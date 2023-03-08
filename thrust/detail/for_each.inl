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


/*! \file for_each.inl
 *  \brief Inline file for for_each.h.
 */

#include <thrust/detail/config.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/for_each.h>
// 这个文件的inclusion，会导入所有系统定义了`for_each`的头文件
#include <thrust/system/detail/adl/for_each.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction>
__host__ __device__
  InputIterator for_each(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         InputIterator first,
                         InputIterator last,
                         UnaryFunction f)
{
  using thrust::system::detail::generic::for_each;
  // 这里采用了ADL (Argument Dependent Lookup)技巧
  // 根据第一个参数的类型定义所在的namespace，找到那个namespace所在空间内定义的`for_each`
  // 如果没找定义，则采用generic::for_each
  // 类似于using std::swap的使用
  return for_each(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, f);
}


template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  // 如果用户没有提供execution policy，自己推导
  using thrust::system::detail::generic::select_system;
  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;
  return thrust::for_each(select_system(system), first, last, f);
} // end for_each()

__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Size, typename UnaryFunction>
__host__ __device__
  InputIterator for_each_n(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator first,
                           Size n,
                           UnaryFunction f)
{
  using thrust::system::detail::generic::for_each_n;

  return for_each_n(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, n, f);
} // end for_each_n()


template<typename InputIterator,
         typename Size,
         typename UnaryFunction>
InputIterator for_each_n(InputIterator first,
                         Size n,
                         UnaryFunction f)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;
  return thrust::for_each_n(select_system(system), first, n, f);
} // end for_each_n()

THRUST_NAMESPACE_END
