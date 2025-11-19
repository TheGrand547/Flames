#version 440 core
#include "lighting"
#include "frustums"
#include "camera"
#include "forward_buffers"
#include "cone"

// GET BACK TO THIS
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#ifndef MAX_LIGHTS
#define MAX_LIGHTS 100
#endif

shared uint numLights;
shared uint groupLights[MAX_LIGHTS];
shared Frustum groupFrustum;
shared uint globalOffset;
shared uint maxDepth;
shared uint minDepth;
shared float clipNear;
shared uint lightBitmask;

uniform sampler2D DepthBuffer;
uniform uint featureToggle;

#ifndef MASKS_PER_TILE
//#define MASKS_PER_TILE 16 + 1
#define MASKS_PER_TILE 17
#endif

shared uint localTileMask[MASKS_PER_TILE];

void AddLight(uint currentIndex)
{
	/*
	uint index = atomicAdd(numLights, 1);
	if (index < MAX_LIGHTS)
	{
		groupLights[index] = currentIndex;
	}
	*/
	uint bucketOffset = currentIndex / 32;
	uint intraOffset  = currentIndex % 32;
	atomicOr(localTileMask[bucketOffset + 1], 1 << intraOffset);
	//atomicAdd(numLights, 1);
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
	LightInfoBig current = lights[index];
	if (SphereTest(current.position, zNear, zFar))
	{
		AddLight(index);
	}
}

void ConeLightCull(uint index, float zNear, float zFar)
{
	LightInfoBig current = lights[index];
	Cone local = LightToCone(current);
	if (CoarseConeTest(local, zNear, zFar) && FrustumCone(groupFrustum, local, zNear, zFar))
	{
		AddLight(index);
	}
}

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;
void main()
{
	const uint groupIndex = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x; 
	uint threadIndex = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x;
	
	float ratio = float(TileSize) / float(BLOCK_SIZE);
	
	vec2 sampleCoord = vec2(gl_WorkGroupID.xy * TileSize + gl_LocalInvocationID.xy * ratio) / ScreenSize;
	sampleCoord.y = 1 - sampleCoord.y;
	float sampledDepth = texture(DepthBuffer, sampleCoord).r;
	uint ordered = floatBitsToUint(sampledDepth);
	if (threadIndex == 0)
	{
		if (groupIndex == 0)
		{
			globalLightIndex = 0;
		}
		minDepth = 0;
		maxDepth = 0xFFFFFFFF;
		numLights = 0;
		globalOffset = 0;
		grid[groupIndex] = uvec2(0, 0);
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
	
	for (; i < lightCount; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		LightInfoBig current = lights[i];
		float type = lights[i].position.w;
		
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
			//AddLight(i);
		}
	}
	barrier();
	// Actually save them here
	if (threadIndex == 0)
	{
		//globalOffset = atomicAdd(globalLightIndex, numLights);
		//grid[groupIndex] = uvec2(globalOffset, numLights);
		//for (uint i = 0; i < numLights; i += 1)
		{
			//indicies[globalOffset + i] = groupLights[i];
		}
		// Tells us directly how many lights are in this tile
		//tileMasks[groupIndex * MASKS_PER_TILE] = numLights;
		uint totalLights = 0;
		for (uint i = 1; i < MASKS_PER_TILE; i++)
		{
			tileMasks[groupIndex * MASKS_PER_TILE + i] = localTileMask[i];
			totalLights += bitCount(localTileMask[i]);
		}
		tileMasks[groupIndex * MASKS_PER_TILE] = totalLights;
	}
	
	// TODO: Possibly return to this
	barrier();
	const uint baseOffset = groupIndex * MASKS_PER_TILE + 1;
	for (uint i = threadIndex + 1; i < MASKS_PER_TILE; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		//indicies[globalOffset + i] = groupLights[i];
		//tileMasks[i] = localTileMask[i];
	}
}