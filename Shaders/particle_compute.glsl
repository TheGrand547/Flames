#version 440 core

#include "camera"

// TODO: Maybe some compression via using halfs and stuff like that, I don't know
// velocity.w will decrease by normal.w until velocity.w < 0, at which point it'll be removed
struct Particle
{
	vec4 position, velocity, normal, color;
};

struct DrawParticle
{
	vec4 position, color;
};

layout(std430, binding = 0) volatile buffer RawParticles
{
	Particle elements[];
};

layout(std430, binding = 1) volatile buffer DrawParticles
{
	DrawParticle drawElements[];
};

layout(std430, binding = 2) volatile buffer IndirectParticles
{
	uint count;
	uint primCount;
	uint first;
	uint baseInstance;
};


layout(std430, binding = 3) readonly buffer NewParticles
{
	Particle newParticles[];
};


// Persistant data about the particles
layout(std430, binding = 4) volatile buffer MiscParticles
{
	uint circularBufferIndex;
	uint newParticleCount;
};

#ifndef POOL_SIZE
#define POOL_SIZE 64
#endif // POOL_SIZE

#ifndef MAX_PARTICLES
#define MAX_PARTICLES 1024
#endif // MAX_PARTICLES

shared uint CopiedElements[POOL_SIZE];

shared uint drawIndex;
shared uint drawCount;

layout(location = 0) uniform uint pauseMotion;

layout(local_size_x = POOL_SIZE, local_size_y = 1, local_size_z = 1) in;
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
	if (groupIndex == 0 && newParticleCount > 0)
	{
		uint baseIndex = circularBufferIndex + globalIndex;
		for (uint i = 0; i < newParticleCount; i += WorkGroupSize)
		{
			uint outIndex = (baseIndex + i) % MAX_PARTICLES;
			Particle current = newParticles[i];
			const float timeDelta = 1.f / 128.f;
			current.velocity.xyz *= timeDelta;
			current.normal *= timeDelta;
			elements[outIndex] = current;
		}
	}
	
	barrier();
	if (globalIndex == 0 && newParticleCount > 0)
	{
		circularBufferIndex = (circularBufferIndex + newParticleCount) % MAX_PARTICLES;
		newParticleCount = 0;
	
	}
	if (globalIndex < elements.length())
	{
		Particle current = elements[globalIndex];
		if (current.velocity.w > 0)
		{
			if (pauseMotion == 0)
			{
				current.velocity -= current.normal;
				current.position.xyz += current.velocity.xyz;
				elements[globalIndex] = current;
			}
			uint myIndex = atomicAdd(drawCount, 1);
			if (myIndex < POOL_SIZE)
			{
				CopiedElements[myIndex] = globalIndex;
			}
		}
	}
	barrier();
	// This is stupidly inefficient
	if (workGroupIndex == 0 && drawCount > 0)
	{
		drawIndex = atomicAdd(primCount, drawCount);
		for (int i =0 ; i < drawCount; i++)
		{
			DrawParticle temp;
			temp.position = elements[CopiedElements[i]].position;
			temp.color = elements[CopiedElements[i]].color;
			drawElements[drawIndex + i] = temp;
		}
	}
}