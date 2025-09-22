#version 440 core

#include "frustums"

#include "camera"

// Screen is broken up into 16x16 TILES
// A 8x8 arrangement of TILES is called to used by this shader to compute the frustums for the TILES

uniform int TileSize;

uniform vec2 ScreenSize;


// Determines, per work group, how many of each local threads there will be there are
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

vec4 TransformToView(vec2 ins)
{
	vec4 temp = inverse(Projection) * vec4(ins / ScreenSize * 2.f - 1.f, -1.f, 1.f);
	temp /= temp.w;
	return temp;
	
};
vec4 TransformToView4(vec4 ins)
{
	vec4 temp = inverse(Projection) * ins;
	temp /= temp.w;
	return temp;
};

void main()
{
	vec2 size = vec2(TileSize, 0);
	vec2 position = size.xx * gl_WorkGroupID.xy;
	if (position.x > ScreenSize.x || position.y > ScreenSize.y)
	{
		return;
	}
	
	vec2 points[4] = {position + size.yy,  position + size.xy, position + size.xx, position + size.yx};
	vec3 points2[4];
	for (int i = 0; i < 4; i++)
	{
		points2[i] = TransformToView(min(points[i], ScreenSize)).xyz;
	}
	const vec3 eye = vec3(0, 0, 0);
	
	Frustum result;
	result.planes[0] = MakePlane(eye, points2[3], points2[0]); // Left
	result.planes[1] = MakePlane(eye, points2[1], points2[2]); // Right
	
	result.planes[2] = MakePlane(eye, points2[0], points2[1]); // Top
	result.planes[3] = MakePlane(eye, points2[2], points2[3]); // Bottom
	
	frustums[gl_LocalInvocationIndex] = result;
};