#version 440 core

layout(location = 0) out vec2 fTex;

uniform vec4 rectangle;
uniform sampler2D image;
uniform int index;

layout(std140) uniform ScreenSpace
{
	mat4 Projection;
};

void main()
{
	int vertexX = 2 - int(index / 3);
	int vertexY = index % 3;
	vec2 pos = rectangle.xy;
	
	vec2 size = rectangle.zw;
	
	pos += ceil((rectangle.zw - size) / 2);
	// TODO: figure out why this is so damn complicated
	
	fTex = vec2(0, 1);
	if ((gl_VertexID % 4) % 2 == 1)
	{
		pos += vec2(0, size.y);
		fTex.y -= 1;
	}
	if (gl_VertexID % 4 > 1)
	{
		pos += vec2(size.x, 0);
		fTex.x += 1;
	}
	fTex += vec2(vertexY, vertexX);
	fTex /= 3;
	fTex.y = 1 - fTex.y;
	gl_Position = Projection * vec4(pos.xy, 0, 1);
}