#version 440 core
#include "lighting"
#include "frustums"
#include "camera"
#include "forward_buffers"
#include "cone"

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif // TILE_SIZE

#ifndef SCREEN_SIZE
#define SCREEN_SIZE vec2(1000)
#endif // SCREEN_SIZE

#ifndef MASKS_PER_TILE
#define MASKS_PER_TILE 17
#endif // MASKS_PER_TILE

#ifndef MAX_LIGHTS
#define MAX_LIGHTS ((MASKS_PER_TILE - 1) * 32)
#endif // MAX_LIGHTS

layout(location = 0) uniform sampler2D DepthBuffer;


shared Frustum groupFrustum;
shared uint maxDepth;
shared uint minDepth;
shared float clipNear;
shared uint lightBitmask;

shared uint localTileMask[MASKS_PER_TILE];

void AddLight(uint currentIndex);
bool SphereBitmaskTest(vec2 bounds, float zNear, float zFar);
bool SphereTest(vec4 sphere, float zNear, float zFar);
bool CoarseConeTest(Cone cone, float zNear, float zFar);
void PointLightCull(uint index, float zNear, float zFar);
void ConeLightCull(uint index, float zNear, float zFar);

layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;
void main()
{
	const uint groupIndex = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x; 
	uint threadIndex = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x;
		
	vec2 sampleCoord = vec2(gl_WorkGroupID.xy * TILE_SIZE + gl_LocalInvocationID.xy) / ScreenSize;
	sampleCoord.y = 1 - sampleCoord.y;
	float sampledDepth = texture(DepthBuffer, sampleCoord).r;
	uint ordered = floatBitsToUint(sampledDepth);
	if (threadIndex == 0)
	{
		minDepth = 0;
		maxDepth = 0xFFFFFFFF;
		groupFrustum = frustums[groupIndex];
		lightBitmask = 0;
		for (uint i = 0; i < MASKS_PER_TILE; i++)
		{
			localTileMask[i] = 0;
		}
	}
	
	float measuredZ = TransformFast(sampledDepth);
	barrier();
	atomicMax(minDepth, ordered);
	atomicMin(maxDepth, ordered);
	barrier();
	
	float zNear = TransformFast(uintBitsToFloat(maxDepth));
	float zFar  = TransformFast(uintBitsToFloat(minDepth));
	
	const uint lightCount = min(largestLights.length(), MAX_LIGHTS);
	
	uint i = threadIndex;
	// This feels backwards, but I can't prove it.
	if (minDepth == floatBitsToUint(0.f))
	{
		i = lightCount + 1;
	}
	
	
	// Now this range is divided into pieces
	// Flipped because of reverse Z
	float depthIntervals = 32.f / (zFar - zNear);
	uint terrainBit = uint(max(0, min(32, floor((measuredZ - zNear) * depthIntervals)))); 
	atomicOr(lightBitmask, 1 << terrainBit);
	barrier();
	
	for (; i < lightCount; i += TILE_SIZE * TILE_SIZE)
	{
		float type = largestLights[i].position.w;
		// Sphereical Point light
		if (type > 0)
		{
			PointLightCull(i, zNear, zFar);
		}
		// Directed point light(cone)
		else if (type < 0)
		{
			ConeLightCull(i, zNear, zFar);
		}
		// Type 0, directed light
		else
		{
			AddLight(i);
		}
	}
	barrier();
	if (threadIndex == 0)
	{
		// Tells us directly how many lights are in this tile
		uint totalLights = 0;
		for (uint i = 1; i < MASKS_PER_TILE; i++)
		{
			tileMasks[groupIndex * MASKS_PER_TILE + i] = localTileMask[i];
			totalLights += bitCount(localTileMask[i]);
		}
		tileMasks[groupIndex * MASKS_PER_TILE] = totalLights;
	}
}

void AddLight(uint currentIndex)
{
	uint bucketOffset = currentIndex / 32;
	uint intraOffset  = currentIndex % 32;
	atomicOr(localTileMask[bucketOffset + 1], 1 << intraOffset);
}

// Bitmask idea from https://wickedengine.net/2018/01/optimizing-tile-based-light-culling/
// True  -> no bitmask overlap, light should not be included
// False -> bitmask overlap, more testing needed
bool SphereBitmaskTest(vec2 bounds, float zNear, float zFar)
{	
	float depthIntervals = 32.f / (zFar - zNear);
	uint lowIndex  = uint(max(0, min(32, floor((bounds.x - zNear) * depthIntervals))));
	uint highIndex = uint(max(0, min(32, floor((bounds.y - zNear) * depthIntervals))));
	uint mask = 0xFFFFFFFF;
	mask >>= 31 - (lowIndex - highIndex);
	mask <<= highIndex;
	
	return (mask & lightBitmask) == 0;
}

// True  -> Intersects the current frustum, in some way that matters
// False -> Doesn't intersect the current frustum in a way that matters
bool SphereTest(vec4 sphere, float zNear, float zFar)
{
	float pointNear = sphere.z + sphere.w;
	float pointFar  = sphere.z - sphere.w;
	
	if (SphereBitmaskTest(vec2(pointNear, pointFar), zNear, zFar) || pointNear < zNear || pointFar > zFar)
	{
		return false;
	}
	else if (FrustumSphere(groupFrustum, sphere))
	{
		return true;
	}
	return false;
}


// True  -> Can't rule out that the cone intersects this tile, by assuming it acts as a sphere
// False -> This cone definitely should not be included
bool CoarseConeTest(Cone cone, float zNear, float zFar)
{
	// If this is unbounded, this should be skipped,  but I haven't determined how I am going to do that
	
	// Generate sphere
	float radius = cone.height * 0.5f / (cone.angle * cone.angle);
	
	vec4 sphere = vec4(cone.position + cone.forward * radius, radius);
	return SphereTest(sphere, zNear, zFar);
}

void PointLightCull(uint index, float zNear, float zFar)
{
	LightInfoBig current = GetViewSpaceLighting(largestLights[index]);
	if (SphereTest(current.position, zNear, zFar))
	{
		AddLight(index);
	}
}

void ConeLightCull(uint index, float zNear, float zFar)
{
	LightInfoBig current = GetViewSpaceLighting(largestLights[index]);;
	Cone local = LightToCone(current);
	if (CoarseConeTest(local, zNear, zFar) && FrustumCone(groupFrustum, local, zNear, zFar))
	{
		AddLight(index);
	}
}