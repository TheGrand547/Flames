#version 440 core

#include "camera"
#include "hash"

struct Oblong
{
	vec4 position, velocity;
};

layout(std430, binding = 0) volatile buffer RawDebris
{
	//vec4 elements[];
	Oblong elements[];
};

layout(std430, binding = 1) volatile buffer DrawDebris
{
	vec4 drawElements[];
};

layout(std430, binding = 2) volatile buffer DebrisIndirect
{
	uint count;
	uint primCount;
	uint first;
	uint baseInstance;
};

#ifndef DEBRIS_COUNT
#define DEBRIS_COUNT 1024
#endif

#define COPY_LENGTH 10
shared uint CopiedElements[COPY_LENGTH];

shared uint drawIndex;
shared uint drawCount;

layout(location = 0) uniform vec3 cameraForward;
layout(location = 1) uniform vec3 cameraPos;
layout(location = 2) uniform vec3 cameraVelocity;
layout(location = 3) uniform float zFar;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{
	const uint WorkGroupSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;	
	// The flattened index of the current work group
	const uint groupIndex = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y;

	// The index of this instance within the current work group
	const uint workGroupIndex = gl_LocalInvocationID.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
	
	// Unique index of this compute instance
	const uint globalIndex = groupIndex * WorkGroupSize + workGroupIndex;
	
	if (globalIndex == 0)
	{
		count = 4; // the number of vertices in the draw call
		first = 0; // First 'index' in the array of 'indices'
		baseInstance = 0;
		primCount = 0;
	}
	if (workGroupIndex == 0)
	{
		drawIndex = 0;
		drawCount = 0;
	}
	barrier();
	if (globalIndex < elements.length())
	{
		Oblong current = elements[globalIndex];
		vec3 currentDust = current.position.xyz;
		// Move dust
		
		// A constant motion to imply that the 'big ship' is moving, albiet slowly
		const vec3 dustDirection = pow(2.f, -7.f) * 0.4f * normalize(vec3(1.f, -0.25f, 0.325f));
		
		
		const float angleDeviation = 50;
		const float cosineValue = cos(radians(angleDeviation));
		
		currentDust += dustDirection + current.velocity.xyz;
		// Since View[3] is negated, the double negative becomes a position(maybe)
		vec3 delta = currentDust - cameraPos;
		float alignment = dot(cameraForward, normalize(delta));
		float distance = length(delta);
		//if ((alignment < -cosineValue && length(delta) > zFar *0.8) || length(delta) > 2 * zFar)
		if (distance > 0.25 * zFar)
		{
			// Regenerate
			vec3 position = hash3D(currentDust + current.velocity.xyz + vec3(alignment)) * 2.f - 1.f;
			position = position * zFar / 4.f + cameraPos + cameraVelocity;
			currentDust = position;
			elements[globalIndex].position.w = max(hash1D(position) / 2.f, 0.15);
		}
		
		
		// Transform dust to camera
		elements[globalIndex].position.xyz = currentDust;
		
		// Only pass particles in front of the camera to the shader
		if (alignment > 0)
		{
			uint myIndex = atomicAdd(drawCount, 1);
			if (myIndex < COPY_LENGTH)
			{
				CopiedElements[myIndex] = globalIndex;
			}
		}
	}
	barrier();
	if (workGroupIndex == 0 && drawCount > 0)
	{
		drawIndex = atomicAdd(primCount, drawCount);
		for (int i =0 ; i < drawCount; i++)
		{
			drawElements[drawIndex + i] = elements[CopiedElements[i]].position;
		}
	}
	
}