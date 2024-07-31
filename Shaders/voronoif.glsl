#version 440 core

layout(location = 0) in vec2 textureCoords;
layout(location = 0) out vec4 color;

// 0 => Distance from closest node
// 1 => Colored based on the closest node
// 2 => Voronoi Edges
uniform int mode;

layout(std140) uniform Points
{
	vec2 points[32];
	int Length;
};

void main()
{
	vec2 uv = textureCoords;
	float minLength = 10;
	vec2 minPoint = points[0];
	int index = 0;
	for (int i = 0; i < Length; i++)
	{
		for (int x = -1; x < 2; x++)
		{
			for (int y = -1; y < 2; y++)
			{
				vec2 local = points[i] + vec2(x, y);
				float estimate = length(uv - local);
				
                if (estimate < minLength)
                {
					color.z = 0;
					index = i;
                    minLength = estimate;
                    minPoint = local;
                }
			}
		}
	}
	float curved = pow(1. - minLength, 6.);
	if (mode == 0)
	{
		color = vec4(curved, curved, curved, 1);
	}
	else if (mode == 1)
	{
		color = vec4(points[index], curved, 1);
	}
	else
	{
		float edgeLength = 2.f;
		vec2 edgePoint = points[0];
		for (int i = 0; i < Length; i++)
		{
			for (int x = -1; x < 2; x++)
			{
				for (int y = -1; y < 2; y++)
				{
					vec2 local = points[i] + vec2(x, y);
					if (minPoint == local)
						continue;
					vec2 avg = uv - ((minPoint + local) / 2.f);
					vec2 dir = normalize(minPoint - local);
					float guess = dot(avg, dir);

					if (guess < edgeLength)
					{
					   edgeLength = guess;
					   edgePoint = local;
					}
				}
			}
		}
		curved = pow(1 - edgeLength, 6); // This gives the really nice curved edges
		color = vec4(curved, curved, curved, 1);
	}
}