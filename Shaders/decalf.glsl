#version 440 core

layout(location = 0) in vec2 fTex;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform sampler2D textureIn;

layout (depth_less) out float gl_FragDepth;

void main()
{
	colorOut = texture(textureIn, fTex);
	normalOut = colorOut;
	gl_FragDepth = gl_FragCoord.z - 0.0001;
}
