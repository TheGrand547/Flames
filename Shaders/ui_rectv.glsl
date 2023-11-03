#version 440 core

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
	if ((gl_VertexID % 4) % 2 == 1)
	{
		pos += vec2(0, delta.y);
	}
	if (gl_VertexID % 4 > 1)
	{
		pos += vec2(delta.x, 0);
	}
	gl_Position = Projection * vec4(pos.xy, 0, 1);
}