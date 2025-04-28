#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in mat3 TBNmat;

layout(location = 0) out vec4 fragmentPosition;
layout(location = 1) out vec4 fragmentNormal;
layout(location = 2) out vec4 fragmentColor;

uniform vec3 shapeColor;
void main()
{
	// TODO: Texture reads for maps and stuff
	//vec3 norm = texture(normalMapIn, samplePoint).rgb;
	//norm = 2 * norm - 1;
	vec3 norm = vec3(0, 0, 1);
	
	// Could also store extra information in those unused 1 segments, idk
	fragmentPosition = vec4(fPos, 1);
	fragmentNormal = vec4(TBNmat * norm, 1);
	fragmentColor = vec4(shapeColor, 1);
}