#version 440 core

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

//uniform vec3 color;
layout(location = 0) in vec4 fColor;

void main()
{
	colorOut = fColor;
	normalOut = fColor;
	//normalOut = vec4(gl_FragCoord.xyz, 1); // Don't want the uniform outline on the lines
}