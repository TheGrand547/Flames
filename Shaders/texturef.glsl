#version 440 core

in vec2 texCoord;

out vec4 fColor;

uniform sampler2D sampler;

void main()
{
	fColor = texture(sampler, texCoord);
}