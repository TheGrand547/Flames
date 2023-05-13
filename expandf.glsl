#version core 440

in vec2 textureCoords;
out vec4 fColor;

uniform sampler2D screen;
uniform int depth;

bool dark = false;

void sample(ivec2 offset)
{
	if (magnitude(vec3(textureOffset(screen, textureCoords, offset))
}

main()
{
	vec4 current = texture(screen, textureCoords);
	for (int i = 1; i < depth && !dark; i++)
	{
		
	}
}