#version 440 core

layout(location = 0) out vec2 textureCoords;

vec2 positions[] = {
	vec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)
};

vec2 uvCoords[] = {
	vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), 
	vec2(0.0f, 1.0f), vec2(1.0f, 1.0f)
};

void main()
{
	gl_Position = vec4(positions[gl_VertexID].xy, 0, 1);
	textureCoords = uvCoords[gl_VertexID];
}