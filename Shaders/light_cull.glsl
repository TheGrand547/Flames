#version 440 core
#include "lighting"
#include "frustums"
#include "camera"


layout(std430, binding = 5) buffer Frustums
{
	Frustum frustums[];
};


layout(std430, binding = 6) buffer LightIndices
{
	uint indices[];
};

layout(std430, binding = 7) buffer LightGrid
{
	uvec2 grid[];
};


layout(std430, binding = 8) buffer LightBlock
{
	uint lightCount;
	// These will be already transformed into view space, for convenience
	LightInfo lights[];
};

layout(std430, binding = 9) buffer LightGrid2
{
	uint globalLightIndex;
};


#define BLOCK_SIZE 16
#define MAX_LIGHTS 100

shared uint numLights;
shared uint groupLights[MAX_LIGHTS];
shared Frustum groupFrustum;

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;
void main()
{
	const uint threadIndex = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x;
	if (threadIndex == 0)
	{
		numLights = 0;
		groupFrustum = frustums[gl_WorkGroupID.x + gl_WorkGroupID.y * gl_WorkGroupID.x];
	}
	groupMemoryBarrier();
	for (uint i = threadIndex; i < lightCount; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		LightInfo current = lights[i];
		if (FrustumSphere(groupFrustum, current.position))
		{
			uint index = atomicAdd(numLights, 1);
			if (index < MAX_LIGHTS)
			{
				groupLights[index] = i;
			}
		}
	}
	groupMemoryBarrier();
	
	// Actually save them here
	if (threadIndex == 0)
	{
		uint index = atomicAdd(globalLightIndex, numLights);
		for (int i = 0; i < numLights; i++)
		{
			indices[index + i] = groupLights[i];
		}
		grid[gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.y] = uvec2(index, numLights);
		
	}
}