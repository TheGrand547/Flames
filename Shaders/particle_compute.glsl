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

/*
layout(std430, binding = 3) readonly buffer NewParticles
{
	Particle newParticles[];
};


// Persistant data about the particles
layout(std430, binding = 4) volatile buffer MiscParticles
{
	uint poolHead, poolTail;
};
*/

#define COPY_LENGTH 10

shared uint CopiedElements[COPY_LENGTH];

shared uint drawIndex;
shared uint drawCount;

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
	/*
	// No existing or current particles, get out of here after setting the indirect draw command up
	//if (poolHead == poolTail && newParticles.length() == 0)
	{
		//return;
	}
	*/
	
	
	if (workGroupIndex == 0)
	{
		drawIndex = 0;
		drawCount = 0;
	}
	barrier();
	if (globalIndex < elements.length())
	{
		Particle current = elements[globalIndex];
		current.velocity -= current.normal;
		current.position += vec4(current.velocity.xyz, 0);
		// Do the dust calculations
		//if (current.velocity.w >= 0)
		{
			uint myIndex = atomicAdd(drawCount, 1);
			if (myIndex < COPY_LENGTH)
			{
				CopiedElements[myIndex] = globalIndex;
			}
		}
		elements[globalIndex] = current;
	}
	barrier();
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