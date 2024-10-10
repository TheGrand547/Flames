#include "AABB.h"
#include "glmHelp.h"
#include "Sphere.h"

AABB& AABB::operator=(const AABB& other) noexcept
{
	this->center = other.center;
	this->halfs = other.halfs;
	return *this;
}

Model AABB::GetModel() const noexcept
{
	// Will return the transform matrix for the unit cube centered at the origin with 2x2x2 dimension
	glm::vec3 transform = this->GetCenter();
	glm::vec3 scale     = this->Deviation();
	return Model(transform, glm::vec3(0), scale);
}


bool AABB::Overlap(const Sphere& other, Collision& collision) const noexcept
{
	glm::vec3 closest = glm::max(this->center - this->halfs, glm::min(other.center, this->center + this->halfs));
	float distance = glm::length(closest - other.center);
	collision.normal = glm::normalize(other.center - closest); // Get vector in the direction of the center from "here"
	if (glm::any(glm::isnan(collision.normal)))
	{
		collision.normal = closest;
	}
	collision.distance = other.radius - distance;
	collision.point = other.center + collision.normal * collision.distance;
	return distance < other.radius;
}

// Modified version of the OBB code to be in theory "better", ie faster
bool AABB::Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& nearHit, Collision& farHit) const noexcept
{
	nearHit.Clear();
	farHit.Clear();
	nearHit.distance = -std::numeric_limits<float>::infinity();
	farHit.distance = std::numeric_limits<float>::infinity();

	glm::vec3 direction = this->center - point;

	for (auto i = 0; i < 3; i++)
	{
		float scale = this->halfs[i];
		float parallel = direction[i];
		if (glm::abs(dir[i]) < EPSILON)
		{
			//if (-parallel - scale > 0 || -parallel + scale > 0)
			if (abs(parallel) > scale)
			{
				return false;
			}
		}

		float scaling = dir[i];
		float param0 = (parallel + scale) / scaling;
		float param1 = (parallel - scale) / scaling;

		if (param0 > param1)
		{
			std::swap(param0, param1);
		}
		if (param0 > nearHit.distance)
		{
			nearHit.distance = param0;
			nearHit.normal = glm::vec3(i == 0, i == 1, i == 2) * glm::sign(-parallel);
		}
		if (param1 < farHit.distance)
		{
			farHit.distance = param1;
			farHit.normal = glm::vec3(i == 0, i == 1, i == 2) * glm::sign(-parallel);
		}
		if (nearHit.distance > farHit.distance)
		{
			return false;
		}
		if (farHit.distance < 0)
		{
			return false;
		}
	}
	nearHit.point = nearHit.distance * dir + point;
	farHit.point = farHit.distance * dir + point;
	if (nearHit.distance < 0)
	{
		std::swap(nearHit, farHit);
	}
	return true;
}

// https://tavianator.com/2022/ray_box_boundary.html#signs maybe?
bool AABB::FastIntersect(const glm::vec3& point, const glm::vec3& dir) const noexcept
{
	static std::size_t failed = 0, succeed = 0;
	float tmin = 0.0, tmax = INFINITY;
	glm::vec3 dir_inv = 1.f / dir;
	glm::vec3 corners[2]{ this->center - halfs, this->center + halfs };

	for (int d = 0; d < 3; ++d) {
		bool sign = signbit(dir_inv[d]);
		float bmin = corners[sign][d];
		float bmax = corners[!sign][d];

		float dmin = (bmin - point[d]) * dir_inv[d];
		float dmax = (bmax - point[d]) * dir_inv[d];

		tmin = glm::max(dmin, tmin);
		tmax = glm::min(dmax, tmax);
	}
	//return tmin < tmax;
	bool res = tmin <= tmax;


	glm::vec3 delta = this->center - point;
	float nearHit = -std::numeric_limits<float>::infinity(), farHit = std::numeric_limits<float>::infinity();
	bool result2 = false;
	for (auto i = 0; i < 3; i++)
	{
		float scale = this->halfs[i];
		float parallel = delta[i];
		if (glm::abs(dir[i]) < EPSILON)
		{
			if (abs(parallel) > scale)
			//if (-parallel - scale > 0 || -parallel + scale > 0)
			{
				//result2 = true;
				//break;
				//return false;
			}
		}

		float scaling = dir[i];
		float param0 = (parallel + scale) / scaling;
		float param1 = (parallel - scale) / scaling;

		if (param0 > param1)
		{
			std::swap(param0, param1);
		}
		if (param0 > nearHit)
		{
			nearHit = param0;
		}
		if (param1 < farHit)
		{
			farHit = param1;
		}
		if (nearHit > farHit)
		{
			//result2 = true;
			//break;
			//return false;
		}
		if (farHit < 0)
		{
			//result2 = true;
			//break;
			//return false;
		}
	}
	result2 = farHit > 0 && nearHit < farHit;
	//return true;
	if (res != result2)
	{
		std::cout << tmin << ":" << tmax << "\t" << nearHit << ":" << farHit << '\n';
		failed++;
	}
	else
	{
		succeed++;
	}
	if (failed % 1000 == 1 || succeed % 1000 == 0)
	{
		std::cout << failed << ":" << succeed << "\n";
	}
	return res;
}