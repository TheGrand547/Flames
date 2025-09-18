#version 440 core

const int BlockSize = 16;

// Determines, per work group, how many of each element there are
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Plane
{
	vec3 normal;
	float distance;
};

struct Frustum
{
	Plane planes[4];
};

/*
layout(std430, binding = 5) buffer Frustums
{
	Frustum frustums[];
}*/

uniform int Width;
uniform int Height; 

layout(std430, binding = 0) buffer Buffered
{
	uint array[256];
};


void main()
{
	array[gl_WorkGroupID.x] = gl_WorkGroupID.x;

};