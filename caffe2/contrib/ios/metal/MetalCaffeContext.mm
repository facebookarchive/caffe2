// Copyright 2004-present Facebook. All Rights Reserved.

#import "MetalCaffeContext.h"
#import "MetalContext.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(TensorMetal);

MetalAllocator* MetalAllocator::singleton_ = NULL;

// Get the Metal Allocator
MetalAllocator* MetalAllocator::Singleton() {
  if (singleton_ == NULL) {
    singleton_ = new MetalAllocator([MetalContext getContext].device);
    CAFFE_ENFORCE(singleton_ != NULL);
  }
  return singleton_;
}

// class MetalAllocator
MetalAllocator::MetalAllocator(id<MTLDevice> device) : device_(device) {
  buffer_cache_ = [NSMutableDictionary<NSNumber *, id<MTLBuffer>> dictionary];
}

MetalAllocator::~MetalAllocator() {
  for (id key in buffer_cache_) {
    id<MTLBuffer> buffer = buffer_cache_[key];
    [buffer_cache_ removeObjectForKey:key];
  }
}

std::pair<void*, MemoryDeleter> MetalAllocator::New(size_t nbytes) {
  id<MTLBuffer> buffer = [device_ newBufferWithLength:nbytes options:MTLResourceCPUCacheModeDefaultCache];
  void *data = [buffer contents];
  NSNumber *key = @((unsigned long long)data);
  buffer_cache_[key] = buffer;
  return {data, Delete};
}

void MetalAllocator::Delete(void *data) {
  NSNumber *key = @((unsigned long long)data);
  id<MTLBuffer> buffer = Singleton()->buffer_cache_[key];
  [Singleton()->buffer_cache_ removeObjectForKey:key];
  buffer = nil;
}

id<MTLBuffer> MetalAllocator::Buffer(void *data) {
  NSNumber *key = @((unsigned long long)data);
  return buffer_cache_[key];
}

// class MetalCaffeContext
std::pair<void*, MemoryDeleter> MetalCaffeContext::New(size_t nbytes) {
  return GetMetalAllocator()->New(nbytes);
}
} // namespace caffe2
