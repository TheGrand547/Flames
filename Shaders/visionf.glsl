#version 440 core

layout(location = 0) in vec2 fTex;
layout(location = 1) in float depth;

layout(location = 0) out vec4 fColor;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

uniform float radius;

void main()
{
	float delta = length(fTex);
	if (delta > 1)
		discard;
	// Use the radius to adjust depth when it works
	float distance = sqrt(1 - delta);
	fColor = vec4(distance, 0, 0, 1);
	
	
	float modified = gl_FragCoord.z + distance * radius;
    gl_FragDepth  = (modified / gl_FragCoord.w) * 0.5 + 0.5;
}