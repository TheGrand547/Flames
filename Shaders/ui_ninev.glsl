#version 440 core

layout(location = 0) in vec4 rectangle;

layout(location = 0) out vec2 fTex;

uniform sampler2D image;
uniform int index;

layout(std140) uniform ScreenSpace
{
	mat4 Projection;
};

vec2 textureLUT[] = {vec2(0, 1), vec2(0, 0), vec2(1, 1), vec2(1, 0)};

void main()
{
	//int index = gl_InstanceID;
	vec2 offset = vec2(index % 3, 2 - int(index / 3));
	int vertexY = 2 - int(index / 3);
	int vertexX = index % 3;
	vec2 pos = rectangle.xy;
	
	vec2 size = rectangle.zw;
	
	pos += ceil((rectangle.zw - size) / 2);
	if ((gl_VertexID % 4) % 2 == 1)
	{
		pos += vec2(0, size.y);
	}
	if (gl_VertexID % 4 > 1)
	{
		pos += vec2(size.x, 0);
	}
	
	fTex = textureLUT[gl_VertexID % 4] + vec2(vertexX, vertexY);
	fTex /= 3;
	gl_Position = Projection * vec4(pos.xy, 0, 1);
}