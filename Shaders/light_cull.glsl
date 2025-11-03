#version 440 core
#include "lighting"
#include "frustums"
#include "camera"
#include "forward_buffers"


// GET BACK TO THIS
#define BLOCK_SIZE 16

#define MAX_LIGHTS 100

shared uint numLights;
shared uint groupLights[MAX_LIGHTS];
shared Frustum groupFrustum;
shared uint globalOffset;
shared uint maxDepth;
shared uint minDepth;
shared float clipNear;

uniform sampler2D DepthBuffer;

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
		minDepth = 0xFFFFFFFF;
		maxDepth = 0;
		numLights = 0;
		globalOffset = 0;
		grid[groupIndex] = uvec2(0, 0);
		groupFrustum = frustums[groupIndex];
		clipNear = TransformFast(0);
	}
	barrier();
	atomicMin(minDepth, ordered);
	atomicMax(maxDepth, ordered);
	barrier();
	
	float zNear = TransformFast(uintBitsToFloat(minDepth));
	float zFar  = TransformFast(uintBitsToFloat(maxDepth));

	Plane nearPlane = {vec3(0, 0, -1), -zNear};
	
	uint i = threadIndex;
	if (minDepth == floatBitsToUint(1.f))
	{
		i = lightCount + 1;
	}
	for (; i < lightCount; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		LightInfoBig current = lights[i];
		
		// Check if light is too near/far, being lenient
		// Could also simplify this by making zNear and skipping the nearPlane culling
		if (current.position.z - current.position.w > zNear || current.position.z + current.position.w < zFar)
		{
			
		}
		else if (FrustumSphere(groupFrustum, current.position))
		{
			//if (!SphereBehindPlane(nearPlane, current.position))
			{
				uint index = atomicAdd(numLights, 1);
				if (index < MAX_LIGHTS)
				{
					groupLights[index] = i;
				}
			}
		}
	}
	barrier();
	// Actually save them here
	if (threadIndex == 0)
	{
		globalOffset = atomicAdd(globalLightIndex, numLights);
		grid[groupIndex] = uvec2(globalOffset, numLights);
		for (uint i = 0; i < numLights; i += 1)
		{
			indicies[globalOffset + i] = groupLights[i];
		}
	}
	// TODO: Possibly return to this
	//barrier();
	//for (uint i = threadIndex; i < numLights; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		//indicies[globalOffset + i] = groupLights[i];
	}
}