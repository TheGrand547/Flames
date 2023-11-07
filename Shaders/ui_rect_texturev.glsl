#version 440 core

layout(location = 0) out vec2 fTex;

uniform vec4 rectangle;

layout(std140) uniform ScreenSpace
{
	mat4 Projection;
};

void main()
{
	vec2 topLeft = rectangle.xy;
	vec2 delta = rectangle.zw;
	
	vec2 pos = topLeft;
	fTex = vec2(0, 1);
	if ((gl_VertexID % 4) % 2 == 1)
	{
		pos += vec2(0, delta.y);
		fTex.y -= 1;
	}
	if (gl_VertexID % 4 > 1)
	{
		pos += vec2(delta.x, 0);
		fTex.x += 1;
	}
	gl_Position = Projection * vec4(pos.xy, 0, 1);
}