#version 440 core

uniform vec4 rectangle;

uniform mat4 screenProjection;

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
	gl_Position = screenProjection * vec4(pos.xy, 0, 1);
}