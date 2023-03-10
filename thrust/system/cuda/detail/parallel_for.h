/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/system/cuda/config.h>

#include <thrust/system/cuda/detail/util.h>
#include <thrust/detail/type_traits/result_of_adaptable_function.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/par_to_seq.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub {

namespace __parallel_for {

  // 并行线程执行(PTX)的policy
  // 描述block线程个数，每个线程处理的数据量
  // 两者的乘积就是每个block处理的数据量
  // 因此只需要描述前两者，编译期确定
  template <int _BLOCK_THREADS,
            int _ITEMS_PER_THREAD = 1>
  struct PtxPolicy
  {
    enum
    {
      BLOCK_THREADS    = _BLOCK_THREADS,
      ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
      ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD,
    };
  };    // struct PtxPolicy

// 每一个并行算法都需要定义这样的Tuning类
// 针对不同的GPU架构描述不同的kernel启动配置选项
// 这里定义的是所有可以真正并行处理算法的Tuning类
// transform/copy/fill/... 都会调用这个parallel_for版本
  template <class Arch, class F>
  struct Tuning;

// 针对sm30的配置
  template <class F>
  struct Tuning<sm30, F>
  {
    typedef PtxPolicy<256, 2> type;
  };


// AgentLauncher仅仅是一个启动启
// 在kernel线程中，真正干活的还是这个Agent的entry device函数
  template <class F,
            class Size>
  struct ParallelForAgent
  {
    template <class Arch>
    struct PtxPlan : Tuning<Arch, F>::type
    {
      typedef Tuning<Arch, F> tuning;
    };
    typedef core::specialize_plan<PtxPlan> ptx_plan;

    enum
    {
      ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD,
      ITEMS_PER_TILE   = ptx_plan::ITEMS_PER_TILE,
      BLOCK_THREADS    = ptx_plan::BLOCK_THREADS
    };

    template <bool IS_FULL_TILE>
    static void    THRUST_DEVICE_FUNCTION
    consume_tile(F    f,
                 Size tile_base,
                 int  items_in_tile)
    {
     // T1 T2 T3 ... Tm | T1 T2 T3 ... Tm | ...
     // d1 d2 d2 ... dm  | d1+m d2+m ... dm+m | ...
     // 一个block里面的线程并行处理tile中连续m个数据(d1, d2, d3, ... dm)
     // 然后m个线程同时移到下一个连续m个数据进行处理
     // ITEM代表当前第几个数据区域，BLOCK_THREADS * ITEM
     // Size idx = BLOCK_THREADS * ITEM + threadIdx.x; => 当前线程处理的tile中的第几个数据
#pragma unroll
      for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
      {
        Size idx = BLOCK_THREADS * ITEM + threadIdx.x;
        if (IS_FULL_TILE || idx < items_in_tile)  // 在tile数据范围之内？？
          f(tile_base + idx); // tile_base + idx => 就是全局的数据索引
      }
    }

  // !!!!! kernel entry函数 !!!!!
    THRUST_AGENT_ENTRY(F     f,
                       Size  num_items,
                       char * /*shmem*/ )
    {
      Size tile_base     = static_cast<Size>(blockIdx.x) * ITEMS_PER_TILE;
      Size num_remaining = num_items - tile_base;
      Size items_in_tile = static_cast<Size>(
          num_remaining < ITEMS_PER_TILE ? num_remaining : ITEMS_PER_TILE);

      // 当前block处理全部的tile？
      // 一个tile是一个block需要处理的用户数据量
      // 每个数据需要应用客户端代码传入的functor函数
      if (items_in_tile == ITEMS_PER_TILE)
      {
        // full tile
        consume_tile<true>(f, tile_base, ITEMS_PER_TILE);
      }
      else
      {
        // partial tile
        consume_tile<false>(f, tile_base, items_in_tile);
      }
    }
  };    // struct ParallelForEagent

  // cuda并行运算执行的引擎，入口
  // parallel_for，tbb也有对应的接口
  // 接受一个任务大小和执行任务的函数，以及一个在哪执行的cudaStream(queue)
  template <class F,
            class Size>
  THRUST_RUNTIME_FUNCTION cudaError_t
  parallel_for(Size         num_items,
               F            f,
               cudaStream_t stream)
  {
    if (num_items == 0)
      return cudaSuccess;
    using core::AgentLauncher;
    using core::AgentPlan;

    bool debug_sync = THRUST_DEBUG_SYNC_FLAG;

    typedef AgentLauncher<ParallelForAgent<F, Size> > parallel_for_agent;
    AgentPlan parallel_for_plan = parallel_for_agent::get_plan(stream);

  // 注意这里的kernel名字定义为"transform::agent"
    parallel_for_agent pfa(parallel_for_plan, num_items, stream, "transform::agent", debug_sync);
    // `f`其实是定义个kernel函数
    pfa.launch(f, num_items);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return cudaSuccess;
  }
}    // __parallel_for

__thrust_exec_check_disable__
template <class Derived,
          class F,
          class Size>
void __host__ __device__
parallel_for(execution_policy<Derived> &policy,
             F                          f,
             Size                       count)
{
  if (count == 0)
    return;

  if (__THRUST_HAS_CUDART__)
  {
    cudaStream_t stream = cuda_cub::stream(policy);
    cudaError_t  status = __parallel_for::parallel_for(count, f, stream);
    cuda_cub::throw_on_error(status, "parallel_for failed");
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    for (Size idx = 0; idx != count; ++idx)
      f(idx);
#endif
  }
}

}    // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
