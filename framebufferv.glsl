#version 440 core

in vec4 positionAndTexture;

out vec2 textureCoords;

void main()
{
	gl_Position = vec4(positionAndTexture.xy, 0, 1);
	textureCoords = positionAndTexture.zw;
}