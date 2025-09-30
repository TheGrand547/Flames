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
	
	float ratio = float(TileSize) / BLOCK_SIZE;
	
	vec2 sampleCoord = vec2(gl_WorkGroupID.xy * TileSize + gl_LocalInvocationID.xy * ratio) / ScreenSize;
	float sampledDepth = texture(DepthBuffer, sampleCoord).r;
	uint ordered = floatBitsToUint(sampledDepth);
	if (threadIndex == 0)
	{
		minDepth = 0xFFFFFFFF;
		maxDepth = 0;
		numLights = 0;
		grid[groupIndex] = uvec2(0, 0);
		groupFrustum = frustums[groupIndex];
	}
	// TODO: try other memory barriers
	groupMemoryBarrier();
	atomicMin(minDepth, ordered);
	atomicMax(maxDepth, ordered);
	groupMemoryBarrier();
	
	float rawNear = uintBitsToFloat(minDepth);
	float rawFar  = uintBitsToFloat(maxDepth);
	
	float zNear   = TransformToView4(vec4(0, 0, rawNear, 1)).z;
	float zFar    = TransformToView4(vec4(0, 0, rawFar, 1)).z;
	Plane nearPlane = Plane(vec3(0, 0, -1), -zNear);
	Plane farPlane = Plane(vec3(0, 0, 1), zFar);
	
	for (uint i = threadIndex; i < lightCount; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		LightInfoBig current = lights[i];
		//if (SphereBehindPlane(nearPlane, current.position) || SphereBehindPlane(farPlane,current.position))
		{
		
		}
		 if (FrustumSphere(groupFrustum, current.position) && !SphereBehindPlane(nearPlane, current.position))// && !SphereBehindPlane(farPlane,current.position))
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
		globalOffset = atomicAdd(globalLightIndex, numLights);
		
		//for (int i = 0; i < numLights; i++)
		{
			//indicies[globalOffset + i] = groupLights[i];
		}
		grid[groupIndex] = uvec2(globalOffset, numLights);
		//grid[groupIndex] = uvec2(gl_WorkGroupID.x, gl_WorkGroupID.y);
	}
	groupMemoryBarrier();
	// TODO: return to this shit
	for (uint i = threadIndex; i < numLights; i += BLOCK_SIZE * BLOCK_SIZE)
	{
		indicies[globalOffset + i] = groupLights[i];
	}
}