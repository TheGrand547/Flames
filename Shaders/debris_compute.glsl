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

shared uint drawIndex;

uniform vec3 cameraForward;
uniform vec3 cameraPos;
uniform float zFar;

layout(local_size_x = DEBRIS_COUNT, local_size_y = 1, local_size_z = 1) in;
void main()
{
	const uint groupIndex = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x; 
	uint threadIndex = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x;
	
	// Group index is me just planning ahead
	if (threadIndex == 0 && groupIndex == 0)
	{
		count = 4; // the number of vertices in the draw call
		first = 0; // First 'index' in the array of 'indices'
		baseInstance = 0;
		primCount = 0;
		drawIndex = 0;
	}
	barrier();
	Oblong current = elements[threadIndex];
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
	if (alignment < -cosineValue && length(delta) > zFar *0.8)
	{
		// Regenerate
		vec3 position = hash3D(currentDust) * 2.f - 1.f;
		
		// Scale the adjustment in the plane of camera forward to be 20% of what it was
		vec3 adjustment = dot(position, cameraForward) * 0.8 * cameraForward;
		position -= adjustment;
		
		// Position is now far in front of 
		position = (cameraForward * zFar * 0.6f) + position * (zFar / 4.f) + cameraPos.xyz;
		currentDust = position;
		elements[threadIndex].position.w = hash1D(position);
	}
	
	
	// Transform dust to camera
	elements[threadIndex].position.xyz = currentDust;
	
	// Only pass particles in front of the camera to the shader
	if (alignment > 0)
	{
		uint myIndex = atomicAdd(drawIndex, 1);
		drawElements[myIndex] = elements[threadIndex].position;
	}		
	barrier();
	if (threadIndex == 0 && groupIndex == 0)
	{
		primCount = drawIndex;
	}
}