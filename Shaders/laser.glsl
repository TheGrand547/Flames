#version 440 core
#include "camera"

#ifdef VERTEX

layout(location = 0) in vec3 vPos;

#ifdef INSTANCED
layout(location = 1) in mat4 Model;
#endif // INSTANCED

void main()
{
#ifdef INSTANCED
	gl_Position = Projection * View * Model * vec4(vPos, 1.f);
#else
	gl_Position = Projection * View  * vec4(vPos, 1.0);
#endif // INSTANCED
}

#endif // VERTEX

#ifdef GEOMETRY

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
layout(location = 0) out float distance;

//layout(location = 1) uniform float thickness;
//layout(location = 2) uniform vec2 viewportSize;

// These values seem to work for now, can be tweaked in the future if so desired
const float thickness = 50.f;
const vec2 viewportSize = vec2(1000.f);


// In debt to https://stackoverflow.com/questions/54686818/glsl-geometry-shader-to-replace-gllinewidth
void main()
{
	vec4 pointA = gl_in[0].gl_Position;
	vec4 pointB = gl_in[1].gl_Position;
	
	
	vec2 direction = normalize(pointB.xy - pointA.xy);
	vec2 normal = direction.yx;
	normal.y *= -1.f;
		
	normal *= (1.f / viewportSize);
	
	vec4 pointAOffset = vec4(normal * (thickness + pointA.w), 0, 0);
	vec4 pointBOffset = vec4(normal * (thickness + pointB.w), 0, 0);	
	
	vec4 coordinates[4];
	coordinates[0] = pointA + pointAOffset;
	coordinates[2] = pointA - pointAOffset;
	
	coordinates[1] = pointB + pointBOffset;
	coordinates[3] = pointB - pointBOffset;
	
	// Number is the index of the point it is based on
	vec4 distances = vec4(0, 1, 0, 1);
	
	
	for (int i = 0 ; i < 4; i++)
	{
		gl_Position = coordinates[i];
		distance = distances[i];
		EmitVertex();
	}
	EndPrimitive();
}

#endif // GEOMETRY

#ifdef FRAGMENT

layout(location = 0) in float distance;

layout(location = 0) out vec4 colorOut;

layout(location = 0) uniform vec4 Color;

void main()
{
	colorOut = Color;
	
#ifndef INSTANCED
	colorOut.w *= max(distance, 0.5);
#endif // INSTANCED
}

#endif // FRAGMENT