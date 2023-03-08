/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{
namespace detail
{


// 这里用的是system::tbb::detail::execution_policy
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
OutputIterator copy(execution_policy<DerivedPolicy> &exec,
                    InputIterator first,
                    InputIterator last,
                    OutputIterator result);


template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
OutputIterator copy_n(execution_policy<DerivedPolicy> &exec,
                      InputIterator first,
                      Size n,
                      OutputIterator result);


} // end namespace detail
} // end namespace tbb
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/tbb/detail/copy.inl>
