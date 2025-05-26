#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in mat3 TBNmat;

layout(location = 0) out vec4 fragmentPosition;
layout(location = 1) out vec4 fragmentNormal;
layout(location = 2) out vec4 fragmentColor;

layout(location = 0) uniform sampler2D textureColor;

uniform vec3 shapeColor;
void main()
{
	// TODO: Texture reads for maps and stuff
	//vec3 norm = texture(normalMapIn, samplePoint).rgb;
	//norm = 2 * norm - 1;
	vec3 norm = vec3(0, 0, 1);
	vec3 mod = (gl_FrontFacing) ? vec3(1.f) : vec3(-1.f);
	
	// Could also store extra information in those unused 1 segments, idk
	fragmentPosition = vec4(fPos, 1);
	
	// This is a hack, but for some reason gl_FrontFacing won't work otherwise. Need to work on this
	fragmentNormal = vec4(TBNmat * norm * mod, 1);
	
	vec4 sampled = texture(textureColor, fTex);
	
	fragmentColor = vec4(shapeColor, 1);
	if (sampled.r != 0)
	{
		fragmentColor = sampled;
	}
}