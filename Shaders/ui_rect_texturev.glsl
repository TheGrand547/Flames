#version 440 core

layout(location = 0) out vec2 fTex;

uniform vec4 rectangle;
uniform sampler2D image;


layout(std140) uniform ScreenSpace
{
	mat4 Projection;
};

void main()
{
	vec2 pos = rectangle.xy;
	
	vec2 size = min(textureSize(image, 0).xy, rectangle.zw);
	
	pos += ceil((rectangle.zw - size) / 2);
	
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
	gl_Position = Projection * vec4(pos.xy, 0, 1);
}