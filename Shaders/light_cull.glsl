#version 440 core
#include "lighting"
#include "frustums"
#include "camera"
#include "forward_buffers"


// GET BACK TO THIS
#define BLOCK_SIZE 1

#define MAX_LIGHTS 100

shared uint numLights;
shared uint groupLights[MAX_LIGHTS];
shared Frustum groupFrustum;

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;
void main()
{
	const uint groupIndex = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x; 
	uint threadIndex = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x;
	//threadIndex = 0;
	if (threadIndex == 0)
	{
		numLights = 0;
		groupFrustum = frustums[groupIndex];
	}
	memoryBarrierShared();
	for (uint i = threadIndex; i < lightCount; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		LightInfoBig current = lights[i];
		if (FrustumSphere(groupFrustum, current.position))
		{
			uint index = atomicAdd(numLights, 1);
			if (index < MAX_LIGHTS)
			{
				groupLights[index] = i;
			}
		}
	}
	memoryBarrierShared();
	
	// Actually save them here
	if (threadIndex == 0)
	{
		uint index = atomicAdd(globalLightIndex, numLights);
		for (int i = 0; i < numLights; i++)
		{
			indicies[index + i] = groupLights[i];
		}
		grid[groupIndex] = uvec2(index, numLights);
		//grid[groupIndex] = uvec2(gl_WorkGroupID.x, gl_WorkGroupID.y);
		
	}
}