#version 440 core

layout (lines) in;
layout (line_strip, max_vertices = 2) out;

layout(location = 0) in vec4 fColor[];
layout(location = 0) out vec4 fColors;

void main()
{
	vec4 delta = vec4(0, 0.25, 0, 0);
	if ((gl_PrimitiveIDIn % 2) == 0)
	{
		delta = vec4(0, 0.25, 0, 0);
	}
	else
	{
		delta = vec4(0,0,0,0);
	}
	
	fColors = fColor[0];
	gl_Position =  gl_in[0].gl_Position + delta;
	EmitVertex();
	
	
	fColors = fColor[1];
	gl_Position =  gl_in[1].gl_Position + delta;
	EmitVertex();
}