#version 440 core

#include "frustums"

#include "camera"
#include "lighting"
#include "forward_buffers"

#ifndef SCREEN_SIZE
#define SCREEN_SIZE vec2(1000)
#endif // SCREEN_SIZE

#ifndef TILE_SIZE
#define TILE_SIZE (16)
#endif // TILE_SIZE

uniform mat4 InverseProjection;


// Determines, per work group, how many of each local threads there will be there are
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

vec4 TransformToView(vec2 ins)
{
	vec2 grump = vec2(ins / ScreenSize);
	// Reverse Z
	vec4 temp = InverseProjection * vec4((vec2(grump.x, 1.f - grump.y) * 2.f) - 1.f, 0.00001f, 1.f);
	temp /= temp.w;
	return temp;
	
};
vec4 TransformToView4(vec4 ins)
{
	vec4 temp = InverseProjection * ins;
	temp /= temp.w;
	return temp;
};



void main()
{
	vec2 size = vec2(TILE_SIZE, 0);
	vec2 position = size.xx * gl_WorkGroupID.xy;
	if (position.x > SCREEN_SIZE.x || position.y > SCREEN_SIZE.y)
	{
		return;
	}
	
	//vec2 points[4] = {position + size.yy,  position + size.xy, position + size.xx, position + size.yx};
	vec2 points[4] = {position + size.yy, position + size.xy, 
					  position + size.yx, position + size.xx};
	vec3 points2[4];
	for (int i = 0; i < 4; i++)
	{
		points2[i] = TransformToView(min(points[i], SCREEN_SIZE)).xyz;
	}
	const vec3 eye = vec3(0, 0, 1.f);
	
	Frustum result;
	result.planes[0] = MakePlane(eye, points2[2], points2[0]); // Left
	result.planes[3] = MakePlane(eye, points2[1], points2[3]); // Right
	
	result.planes[2] = MakePlane(eye, points2[0], points2[1]); // Top
	result.planes[1] = MakePlane(eye, points2[3], points2[2]); // Bottom
	
	uint index = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
	
	frustums[index] = result;
};