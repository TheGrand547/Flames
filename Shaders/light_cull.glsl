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

void AddLight(uint currentIndex)
{
	uint index = atomicAdd(numLights, 1);
	if (index < MAX_LIGHTS)
	{
		groupLights[index] = currentIndex;
	}
}

// Bitmask idea from https://wickedengine.net/2018/01/optimizing-tile-based-light-culling/
void PointLightCull(uint index, float zNear, float zFar)
{
	LightInfoBig current = lights[index];
	
	float pointNear = current.position.z + current.position.w;
	float pointFar  = current.position.z - current.position.w;
		
	float depthIntervals = 32.f / (zFar - zNear);
	uint lowIndex  = uint(max(0, min(32, floor((pointNear - zNear) * depthIntervals))));
	uint highIndex = uint(max(0, min(32, floor((pointFar - zNear) * depthIntervals))));
	uint mask = 0xFFFFFFFF;
	mask >>= 31 - (lowIndex - highIndex);
	mask <<= highIndex;
	bool fallThrough = true;
	bool bitmaskCheck = (mask & lightBitmask) == 0;
	if (featureToggle > 0)
	{
		fallThrough = (mask & lightBitmask) != 0;
	}
	
	if (!fallThrough || pointNear < zNear || pointFar > zFar)
	{
		
	}
	else if (fallThrough && FrustumSphere(groupFrustum, current.position))
	{
		AddLight(index);
	}
}

void ConeLightCull(uint index, float zNear, float zFar)
{
	LightInfoBig current = lights[index];
	Cone local = LightToCone(current);
	if (FrustumCone(groupFrustum, local, zNear, zFar))
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
			//ConeLightCull(i, zNear, zFar);
			//AddLight(i);
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
		globalOffset = atomicAdd(globalLightIndex, numLights);
		grid[groupIndex] = uvec2(globalOffset, numLights);
		//for (uint i = 0; i < numLights; i += 1)
		{
			//indicies[globalOffset + i] = groupLights[i];
		}
	}
	// TODO: Possibly return to this
	barrier();
	for (uint i = threadIndex; i < numLights; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		indicies[globalOffset + i] = groupLights[i];
	}
}