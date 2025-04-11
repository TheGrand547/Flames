#pragma once
#ifndef PARALLEL_H
#define PARALLEL_H
#include <algorithm>
#include <execution>
#include <utility>


namespace Parallel
{
	bool Enabled() noexcept;
	void SetStatus(bool status) noexcept;

	template<typename ExecutionType, typename Iterator, typename Predicate>
		requires std::is_execution_policy_v<ExecutionType>
	void for_each(ExecutionType execution, Iterator first, Iterator last, Predicate predicate)
	{
		if (::Parallel::Enabled())
		{
			std::for_each(execution, std::forward<Iterator>(first), std::forward<Iterator>(last), predicate);
		}
		else
		{
			std::for_each(std::forward<Iterator>(first), std::forward<Iterator>(last), predicate);
		}
	}

	template<typename ExecutionType, typename Iterator, typename Predicate>
		requires std::is_execution_policy_v<ExecutionType>
	Iterator remove_if(ExecutionType execution, Iterator first, Iterator last, Predicate predicate)
	{
		if (::Parallel::Enabled())
		{
			return std::remove_if(execution, std::forward<Iterator>(first), std::forward<Iterator>(last), predicate);
		}
		else
		{
			return std::remove_if(std::forward<Iterator>(first), std::forward<Iterator>(last), predicate);
		}
	}

	template<typename ExecutionType, typename Type, typename Predicate>
		requires std::is_execution_policy_v<ExecutionType>
	Type::size_type erase_if(ExecutionType execution, Type& container, Predicate predicate)
	{
		if (::Parallel::Enabled())
		{
			// This has had some errors for unclear reasons, might have been to do with closing during the parallel loop
			// but I'm not sure
			auto last = std::remove_if(execution, container.begin(), container.end(), predicate);
			auto difference = container.end() - last;
			container.erase(last, container.end());
			return difference;
		}
		else
		{
			return std::erase_if(container, predicate);
		}
	}
}

#endif // PARALLEL_H