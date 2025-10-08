#version 440 core
#include "lighting"
#include "frustums"
#include "camera"
#include "forward_buffers"


// GET BACK TO THIS
#define BLOCK_SIZE 32

#define MAX_LIGHTS 100

shared uint numLights;
shared uint groupLights[MAX_LIGHTS];
shared Frustum groupFrustum;
shared uint globalOffset;
shared uint maxDepth;
shared uint minDepth;


uniform sampler2D DepthBuffer;
uniform vec2 ScreenSize;
uniform mat4 InverseProjection;
uniform int TileSize;


vec4 TransformToView4(vec4 ins)
{
	vec4 temp = InverseProjection * ins;
	temp /= temp.w;
	return temp;
};

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;
void main()
{
	const uint groupIndex = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x; 
	uint threadIndex = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x;
	
	float ratio = float(TileSize) / float(BLOCK_SIZE);
	//ratio = 1.f;
	
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
	}
	// TODO: try other memory barriers
	//memoryBarrierShared();
	//groupMemoryBarrier();
	barrier();
	atomicMin(minDepth, ordered);
	atomicMax(maxDepth, ordered);
	//memoryBarrierShared();
	//groupMemoryBarrier();
	barrier();
	
	float rawNear = uintBitsToFloat(minDepth);
	float rawFar  = uintBitsToFloat(maxDepth);
	
	float zNear    = TransformToView4(vec4(0, 0, rawNear, 1)).z;
	float zFar     = TransformToView4(vec4(0, 0, rawFar, 1)).z;
	float clipNear = TransformToView4(vec4(0, 0, 0, 1)).z;
	float clipFar  = TransformToView4(vec4(0, 0, 1, 1)).z;

	// I have no clue and have lost multiple hours trying to figure out why this shit doesn't work so I will just have to not touch it
	Plane nearPlane;
	nearPlane.normal = vec3(0, 0, -1);
	nearPlane.distance = -zNear;
	
	uint i = threadIndex;
	//if (zNear == clipFar)
	if (minDepth == floatBitsToUint(1.f))
	{
		i = lightCount + 1;
		//return;
	}
	for (; i < lightCount; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		LightInfoBig current = lights[i];
		
		// Check if light is too near/far, being lenient
		if (current.position.z - current.position.w > clipNear || current.position.z + current.position.w < zFar)
		{
			
		}
		/*
		else if (minDepth == floatBitsToUint(1.f))
		{
		
		}*/
		else if (FrustumSphere(groupFrustum, current.position))
		{
			// This still doesn't work for some reason and I have spent too long today fixing bullshit
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
	//groupMemoryBarrier();
	//memoryBarrierShared();
	barrier();
	// Actually save them here
	if (threadIndex == 0)
	{
		globalOffset = atomicAdd(globalLightIndex, numLights);
		
		//for (int i = 0; i < numLights; i++)
		{
			//indicies[globalOffset + i] = groupLights[i];
		}
		grid[groupIndex] = uvec2(globalOffset, numLights);
		//grid[groupIndex] = uvec2(gl_WorkGroupID.x, gl_WorkGroupID.y);
	}
	//groupMemoryBarrier();
	//memoryBarrierShared();
	barrier();
	// TODO: return to this shit
	for (uint i = threadIndex; i < numLights; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		indicies[globalOffset + i] = groupLights[i];
	}
	groupMemoryBarrier();
	memoryBarrierShared();
}