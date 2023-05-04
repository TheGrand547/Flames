#version 440 core

in vec2 textureCoords;
out vec4 fColor;

uniform sampler2D screen;

void main()
{
	fColor = texture(screen, textureCoords);
}