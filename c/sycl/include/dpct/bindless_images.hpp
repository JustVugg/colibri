//===--- bindless_images.hpp ----------------------*- C++ -*---------------===//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BINDLESS_IMAGE_HPP__
#define __DPCT_BINDLESS_IMAGE_HPP__

#ifdef _WIN32
// DirectX headers
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#endif // _WIN32

namespace dpct {
namespace experimental {

#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES

/// The wrapper structure for external memory resource handle desc
class external_mem_handle_desc {
#ifdef _WIN32
  void *handle;
  const void *name;
#else
  int handle;
#endif // _WIN32
  sycl::ext::oneapi::experimental::external_mem_handle_type handle_type;
  size_t size_in_bytes;
  int flags;

public:
  external_mem_handle_desc() {
#ifdef _WIN32
    handle = nullptr;
    name = nullptr;
#else
    handle = -1;
#endif // _WIN32
    handle_type =
        sycl::ext::oneapi::experimental::external_mem_handle_type::opaque_fd;
    size_in_bytes = 0;
    flags = 0;
  }

#ifdef _WIN32
  /// handle setter
  /// \param [in] win32_handle The win32 NT handle of the external memory
  /// resource
  void set_win32_handle(void *win32_handle) { handle = win32_handle; }

  /// name setter
  /// \param [in] win32_obj_name The object of the external memory resource
  void set_win32_obj_name(const void *win32_obj_name) { name = win32_obj_name; }
#else
  /// fd setter
  /// \param [in] fd The file descriptor of the external memory resource
  void set_fd_handle(int fd) { handle = fd; }
#endif // _WIN32

  /// handle_type setter
  /// \param [in] type The type of the external memory resource handle
  void set_handle_type(
      sycl::ext::oneapi::experimental::external_mem_handle_type type) {
    handle_type = type;
  }

  /// size setter
  /// \param [in] size The size (in bytes) of the external memory resource
  void set_res_size(unsigned long long size) { size_in_bytes = size; }

  /// flags setter
  /// \param [in] flags The flags used to handle importing of external memory
  /// resource
  void set_flags(int flags) { this->flags = flags; }

#ifdef _WIN32
  /// handle getter
  /// \param [out] handle The win32 NT handle of the external memory resource
  void *get_win32_handle() { return handle; }

  /// name getter
  /// \param [out] name The object of the external memory resource
  const void *get_win32_obj_name() { return name; }
#else
  /// fd getter
  /// \param [out] fd The file descriptor of the external memory resource
  int get_fd_handle() { return handle; }
#endif // _WIN32

  /// handle_type getter
  /// \param [out] handle_type The type of the external memory resource handle
  sycl::ext::oneapi::experimental::external_mem_handle_type get_handle_type() {
    return handle_type;
  }

  /// size getter
  /// \param [out] size The size (in bytes) of the external memory resource
  unsigned long long get_res_size() { return size_in_bytes; }

  /// flags getter
  /// \param [out] flags The flags used to handle importing of external memory
  /// resource
  int get_flags() { return flags; }
};

/// The wrapper structure for external image memory desc
class external_mem_img_desc {
  sycl::range<3> size;
  image_channel channel;
  unsigned int num_levels;
  sycl::ext::oneapi::experimental::image_type type;

public:
  external_mem_img_desc() {
    size = sycl::range<3>{0, 0, 0};
    channel = image_channel::create<float>();
    num_levels = 1;
    type = sycl::ext::oneapi::experimental::image_type::standard;
  }

  /// size setter
  /// \param [in] img_dims The dimensions of the imported image resource
  void set_size(sycl::range<3> img_dims) {
    size = img_dims;

    for (int i = 0; i < 3; i++) {
      if (size[i] == 1) {
        size[i] = 0;
      }
    }
  }

  /// image_channel setter
  /// \param [in] img_ch The channel info of the imported image resource
  void set_image_channel(image_channel img_ch) { channel = img_ch; }

  /// num_levels setter
  /// \param [in] numLevels The no. of levels in the imported image resource
  void set_num_levels(unsigned int num_levels) {
    this->num_levels = num_levels;

    if (num_levels > 1) {
      set_image_type(sycl::ext::oneapi::experimental::image_type::mipmap);
    }
  }

  /// type setter
  /// \param [in] img_type The type of the imported image resource
  void set_image_type(sycl::ext::oneapi::experimental::image_type img_type) {
    type = img_type;
  }

  /// size getter
  /// \param [out] size The dimensions of the imported image resource
  sycl::range<3> get_size() { return size; }

  /// image_channel getter
  /// \param [out] image_channel The channel info of the imported image resource
  image_channel get_image_channel() { return channel; }

  /// num_levels getter
  /// \param [out] num_levels The no. of levels in the imported image resource
  unsigned int get_num_levels() { return num_levels; }

  /// type getter
  /// \param [out] type The image type of the imported image resource
  sycl::ext::oneapi::experimental::image_type get_image_type() { return type; }
};

/// The wrapper structure for external buffer memory desc
class external_mem_buf_desc {
  uint64_t size_in_bytes;
  uint64_t offset;
  unsigned int flags;

public:
  external_mem_buf_desc() {
    size_in_bytes = 0;
    offset = 0;
    flags = 0;
  }

  /// size setter
  /// \param [in] size The size (in bytes) of the external buffer resource
  void set_res_size(uint64_t size) { size_in_bytes = size; }

  /// offset setter
  /// \param [in] offset The memory offset of the external buffer resource
  void set_mem_offset(uint64_t offset) { this->offset = offset; }

  /// flags setter
  /// \param [in] flags The flags used to import external buffer resource
  void set_flags(unsigned int flags) { this->flags = flags; }

  /// size getter
  /// \param [out] size_in_bytes The size (in bytes) of the external buffer res
  uint64_t get_res_size() { return size_in_bytes; }

  /// offset getter
  /// \param [out] offset The memory offset of the external buffer resource
  uint64_t get_mem_offset() { return offset; }

  /// flags getter
  /// \param [out] flags The flags used to import external buffer resource
  unsigned int get_flags() { return flags; }
};

/// The wrapper class for external semaphore resource handle desc
class external_sem_handle_desc {
#ifdef _WIN32
  void *handle;
  const void *name;
#else
  int handle;
#endif // _WIN32
  sycl::ext::oneapi::experimental::external_semaphore_handle_type handle_type;
  int flags;

public:
  external_sem_handle_desc() {
#ifdef _WIN32
    handle = nullptr;
    name = nullptr;
#else
    handle = -1;
#endif // _WIN32
    handle_type = sycl::ext::oneapi::experimental::
        external_semaphore_handle_type::opaque_fd;
    flags = 0;
  }

#ifdef _WIN32
  /// handle setter
  /// \param [in] win32_handle The win32 NT handle of the external semaphore
  /// resource
  void set_win32_handle(void *win32_handle) { handle = win32_handle; }

  /// name setter
  /// \param [in] win32_obj_name The object of the external semaphore resource
  void set_win32_obj_name(const void *win32_obj_name) { name = win32_obj_name; }
#else
  /// fd setter
  /// \param [in] fd The file descriptor of the external semaphore resource
  void set_fd_handle(int fd) { handle = fd; }
#endif // _WIN32

  /// handle_type setter
  /// \param [in] type The type of the external semaphore resource handle
  void set_handle_type(
      sycl::ext::oneapi::experimental::external_semaphore_handle_type type) {
    handle_type = type;
  }

  /// flags setter
  /// \param [in] flags The flags used to handle importing of external semaphore
  /// resource
  void set_flags(int flags) { this->flags = flags; }

#ifdef _WIN32
  /// handle getter
  /// \param [out] handle The win32 NT handle of the external semaphore resource
  void *get_win32_handle() { return handle; }

  /// name getter
  /// \param [out] name The object of the external semaphore resource
  const void *get_win32_obj_name() { return name; }
#else
  /// fd getter
  /// \param [out] fd The file descriptor of the external semaphore resource
  int get_fd_handle() { return handle; }
#endif // _WIN32

  /// handle_type getter
  /// \param [out] handle_type The type of the external semaphore resource
  /// handle
  sycl::ext::oneapi::experimental::external_semaphore_handle_type
  get_handle_type() {
    return handle_type;
  }

  /// flags getter
  /// \param [out] flags The flags used to handle importing of external
  /// semaphore
  int get_flags() { return flags; }
};

/// The wrapper structure for external semaphore signal/wait params
class external_sem_params {
  uint64_t value;

public:
  external_sem_params() { value = 0; }

  /// value setter
  /// \param [in] val The state value of the semaphore
  void set_value(unsigned long long val) { value = val; }

  /// value getter
  /// \param [out] value The state value of the semaphore
  unsigned long long get_value() { return value; }
};

/// The wrapper class of external semaphore
class external_sem_wrapper {
  sycl::ext::oneapi::experimental::external_semaphore ext_sem;
  sycl::ext::oneapi::experimental::external_semaphore_handle_type
      ext_sem_handle_type;

public:
  /// Create external semaphore wrapper.
  /// \param [in] ext_sem The external semaphore resource object
  /// \param [in] ext_sem_handle_type The external semaphore resource handle
  /// type
  external_sem_wrapper(
      sycl::ext::oneapi::experimental::external_semaphore ext_sem,
      sycl::ext::oneapi::experimental::external_semaphore_handle_type
          ext_sem_handle_type) {
    this->ext_sem = ext_sem;
    this->ext_sem_handle_type = ext_sem_handle_type;
  }

  /// destory external semaphore resource
  ~external_sem_wrapper() {
    sycl::ext::oneapi::experimental::release_external_semaphore(
        ext_sem, get_default_queue());
  }

  /// ext_sem getter
  /// \param [out] ext_sem The external semaphore resource object
  sycl::ext::oneapi::experimental::external_semaphore get() { return ext_sem; }

  /// ext_sem_handle_type getter
  /// \param [out] ext_sem_handle_type The external semaphore resource handle
  /// type
  sycl::ext::oneapi::experimental::external_semaphore_handle_type
  get_handle_type() {
    return ext_sem_handle_type;
  }
};

/// The wrapper class of bindless image memory handle.
class image_mem_wrapper {
public:
  /// Create bindless image memory wrapper.
  /// \tparam dimensions The dimensions of image memory.
  /// \param [in] channel The image channel used to create bindless image
  /// \param [in] range The sizes of each dimension of bindless image memory.
  /// memory.
  template <int dimensions = 3>
  image_mem_wrapper(image_channel channel, sycl::range<dimensions> range,
                    sycl::ext::oneapi::experimental::image_type type =
                        sycl::ext::oneapi::experimental::image_type::standard,
                    unsigned int num_levels = 1)
      : _channel(channel) {
    unsigned array_size = 1;
    if (type == sycl::ext::oneapi::experimental::image_type::array) {
      assert(dimensions == 3);
      array_size = range[2];
      range[2] = 0;
    }
    _desc = sycl::ext::oneapi::experimental::image_descriptor(
        range, _channel.get_channel_num(), _channel.get_channel_type(), type,
        num_levels, array_size);
    auto q = get_default_queue();
    _handle = alloc_image_mem(_desc, q);
    init_mip_level_wrappers(q);
  }
  /// Create bindless image memory wrapper.
  /// \param [in] channel The image channel used to create bindless image
  /// \param [in] width The width of bindless image memory.
  /// \param [in] height The height of bindless image memory.
  /// \param [in] depth The depth of bindless image memory.
  image_mem_wrapper(image_channel channel, size_t width, size_t height = 0,
                    size_t depth = 0)
      : image_mem_wrapper(channel, {width, height, depth}) {}
  /// Create bindless image memory wrapper.
  /// \param [in] desc The image descriptor used to create bindless image
  image_mem_wrapper(
      const sycl::ext::oneapi::experimental::image_descriptor *desc)
      : _desc(*desc) {
    _channel.set_channel_type(desc->channel_type);
    _channel.set_channel_num(desc->num_channels);
    auto q = get_default_queue();
    _handle = alloc_image_mem(_desc, q);
  }
  /// Create bindless image memory wrapper.
  /// \param [in] desc The image descriptor of bindless image.
  /// \param [in] img_mem_handle The memory handle of bindless image.
  image_mem_wrapper(
      sycl::ext::oneapi::experimental::image_descriptor desc,
      sycl::ext::oneapi::experimental::image_mem_handle img_mem_handle)
      : _desc(desc), _handle(img_mem_handle) {
    _channel.set_channel_type(_desc.channel_type);
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION >= 20240725)
    _channel.set_channel_num(_desc.num_channels);
#endif
    auto q = get_default_queue();
    init_mip_level_wrappers(q);
  }
  /// Create bindless image memory wrapper.
  /// \param [in] extMem SYCL external memory resource object.
  /// \param [in] ext_img_mem_desc The img desc of the imported image resource.
  image_mem_wrapper(sycl::ext::oneapi::experimental::external_mem extMem,
                    external_mem_img_desc *ext_img_mem_desc) {
    auto q = get_default_queue();
    unsigned int array_size = 1;

    _channel = ext_img_mem_desc->get_image_channel();
    _desc = sycl::ext::oneapi::experimental::image_descriptor(
        ext_img_mem_desc->get_size(), _channel.get_channel_num(),
        _channel.get_channel_type(), ext_img_mem_desc->get_image_type(),
        ext_img_mem_desc->get_num_levels(), array_size);
    _handle = sycl::ext::oneapi::experimental::map_external_image_memory(
        extMem, _desc, q);

    init_mip_level_wrappers(q);
  }
  /// Create bindless image memory wrapper.
  /// \param [in] desc The image descriptor of bindless image.
  /// \param [in] num_levels Number of mipmap levels to allocate
  image_mem_wrapper(sycl::ext::oneapi::experimental::image_descriptor desc,
                    unsigned int num_levels)
      : _desc(desc) {
    _desc.type = sycl::ext::oneapi::experimental::image_type::mipmap;
    _desc.num_levels = num_levels;
    auto q = get_default_queue();
    _handle = alloc_image_mem(_desc, q);
    init_mip_level_wrappers(q);
  }

  image_mem_wrapper(const image_mem_wrapper &) = delete;
  image_mem_wrapper &operator=(const image_mem_wrapper &) = delete;
  /// Destroy bindless image memory wrapper.
  ~image_mem_wrapper() {
    if (_sub_wrappers) {
      std::destroy_n(_sub_wrappers, _desc.num_levels);
      std::free(_sub_wrappers);
    }
    free_image_mem(_handle, _desc.type, get_default_queue());
  }
  /// Get the image channel of the bindless image memory.
  /// \returns The image channel of bindless image memory.
  image_channel get_channel() const noexcept { return _channel; }
  /// Get the sizes of each dimension of the bindless image memory.
  /// \returns The sizes of each dimension of bindless image memory.
  sycl::range<3> get_range() const {
    return {_desc.width, _desc.height, _desc.depth};
  }
  /// Get the image descriptor of the bindless image memory.
  /// \returns The image descriptor of bindless image memory.
  const sycl::ext::oneapi::experimental::image_descriptor &
  get_desc() const noexcept {
    return _desc;
  }
  /// Get the image handle of the bindless image memory.
  /// \returns The image handle of bindless image memory.
  sycl::ext::oneapi::experimental::image_mem_handle
  get_handle() const noexcept {
    return _handle;
  }
  /// Get the image mip level of the bindless image memory.
  /// \returns The image mip level of the bindless image memory.
  image_mem_wrapper *get_mip_level(unsigned int level) {
    if (_desc.type == sycl::ext::oneapi::experimental::image_type::standard) {
      assert(level == 0);
      return this;
    }

    assert(_desc.type == sycl::ext::oneapi::experimental::image_type::mipmap);
    return _sub_wrappers + level;
  }

  sycl::ext::oneapi::experimental::image_type get_image_type(void) {
    return _desc.type;
  }

private:
  image_mem_wrapper(
      const image_channel &channel,
      const sycl::ext::oneapi::experimental::image_descriptor &desc,
      const sycl::ext::oneapi::experimental::image_mem_handle &handle)
      : _channel(channel), _desc(desc), _handle(handle) {}

  /// Initialize image mem wrappers for all the levels of mipmap image
  /// \param [in] q The Queue to be used to query the mip levels
  void init_mip_level_wrappers(sycl::queue q) {
    auto num_levels = _desc.num_levels;
    if (_desc.type == sycl::ext::oneapi::experimental::image_type::mipmap) {
      assert(num_levels > 1);
      _sub_wrappers = (image_mem_wrapper *)std::malloc(
          sizeof(image_mem_wrapper) * num_levels);
      for (unsigned i = 0; i < num_levels; ++i)
        new (_sub_wrappers + i) image_mem_wrapper(
            _channel, _desc.get_mip_level_desc(i),
            sycl::ext::oneapi::experimental::get_mip_level_mem_handle(
                _handle, i, q.get_device(), q.get_context()));
    }
  }

  image_channel _channel;
  sycl::ext::oneapi::experimental::image_descriptor _desc;
  sycl::ext::oneapi::experimental::image_mem_handle _handle;
  image_mem_wrapper *_sub_wrappers{nullptr};
};

/// The base wrapper class of external memory resource handle.
class external_mem_wrapper_base {
public:
  /// Cleans up the external mem wrapper by releasing the SYCL external
  /// memory handle
  virtual ~external_mem_wrapper_base() {}

  external_mem_wrapper_base(const external_mem_wrapper_base &) = delete;

  external_mem_wrapper_base &
  operator=(const external_mem_wrapper_base &) = delete;

  const sycl::ext::oneapi::experimental::external_mem
  get_external_mem() const noexcept {
    return _res_external_mem;
  }

  /// Map the buf/texture resource memory and get its pointer/mem handle.
  /// \param [in] q The queue used to map the resource with.
  void map_resource(sycl::queue q) {
    if (_res_is_buffer) {
      if (_res_buf_ptr) {
        throw std::runtime_error("Resource is already mapped! "
                                 "Resource cannot be mapped more than once.");
      }

#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION >= 20240823)
      _res_buf_ptr =
          sycl::ext::oneapi::experimental::map_external_linear_memory(
              get_external_mem(), 0, _res_size_bytes, q);
#else
      throw std::runtime_error(
          "Mapping a buffer resoure is not yet supported!");
#endif
    } else {
      if (_res_img_mem_wrapper_ptr) {
        throw std::runtime_error("Resource is already mapped! "
                                 "Resource cannot be mapped more than once.");
      }

      auto img_mem_handle =
          sycl::ext::oneapi::experimental::map_external_image_memory(
              get_external_mem(), _res_img_desc, q);

      _res_img_mem_wrapper_ptr =
          new image_mem_wrapper(_res_img_desc, img_mem_handle);
    }
  }

  /// Gets the pointer and size of the mapped resource
  /// \param [out] ptr The pointer to the resource memory.
  /// \param [out] size The size of the resource memory.
  /// and resource size.
  void get_mapped_pointer(void **ptr, size_t *size) {
    if (!_res_is_buffer) {
      throw std::runtime_error(
          "Non buffer resouce cannot be accessed as a pointer!");
    }

    *ptr = _res_buf_ptr;
    if (!*ptr) {
      throw std::runtime_error(
          "Resource is not mapped! "
          "Resource should be mapped before accessing its memory");
    }

    *size = _res_size_bytes;
  }

  /// Returns mapped memory as image_mem_wrapper*
  image_mem_wrapper *get_mapped_mipmapped_array() {
    if (_res_is_buffer) {
      throw std::runtime_error(
          "Buffer resource cannot be accessed as an array!");
    }
    if (!_res_img_mem_wrapper_ptr) {
      throw std::runtime_error(
          "Resource is not mapped! "
          "Resource should be mapped before accessing its memory.");
    }

    return _res_img_mem_wrapper_ptr;
  }

  /// Returns mapped subresource memory as image_mem_wrapper*
  /// \param [in] array_index The layer of the image array to be returned.
  /// \param [in] mip_level The level of the mipmap image to be returned.
  image_mem_wrapper *get_sub_resource_mapped_array(unsigned int array_index,
                                                   unsigned int mip_level) {
    if (_res_is_buffer) {
      throw std::runtime_error(
          "Buffer resouce cannot be accessed as an array!");
    }
    if (array_index != 0) {
      throw std::runtime_error("Retrieving a particular layer from image array "
                               "is not yet supported!");
    }
    if (array_index >= _res_img_desc.array_size) {
      throw std::runtime_error(
          "Array index requested exceeds total array size in resource!");
    }
    if (mip_level >= _res_img_desc.num_levels) {
      throw std::runtime_error(
          "Mip level requested exceeds total num of levels in resource!");
    }
    if (!_res_img_mem_wrapper_ptr) {
      throw std::runtime_error(
          "Resouce is not mapped! "
          "Resource should be mapped before accessing its memory");
    }

    return (mip_level == 0)
               ? _res_img_mem_wrapper_ptr
               : _res_img_mem_wrapper_ptr->get_mip_level(mip_level);
  }

  /// Unmap the buf/texture resource memory and get its pointer/mem handle.
  /// \param [in] q The queue used to unmap the resource with.
  void unmap_resource(sycl::queue q) {
    if (_res_is_buffer) {
      if (!_res_buf_ptr) {
        throw std::runtime_error("Resource is not mapped! "
                                 "Resource should be mapped before unmapping.");
      }

      _res_buf_ptr = nullptr;

      throw std::runtime_error(
          "Unmapping a buffer resoure is not yet supported!");
    } else {
      if (!_res_img_mem_wrapper_ptr) {
        throw std::runtime_error("Resource is not mapped! "
                                 "Resource should be mapped before unmapping.");
      }

      delete _res_img_mem_wrapper_ptr;
      _res_img_mem_wrapper_ptr = nullptr;
    }
  }

protected:
  /// Init external mem handle for a given resource.
  external_mem_wrapper_base(
      sycl::ext::oneapi::experimental::external_mem external_mem,
      unsigned reg_flags)
      : _res_external_mem(external_mem), _res_reg_flags(reg_flags) {}

  unsigned _res_reg_flags = 0;
  bool _res_is_buffer = false;
  size_t _res_size_bytes = 0;

  sycl::ext::oneapi::experimental::image_descriptor _res_img_desc;
  sycl::ext::oneapi::experimental::external_mem _res_external_mem;

private:
  void *_res_buf_ptr = nullptr;
  image_mem_wrapper *_res_img_mem_wrapper_ptr = nullptr;
};

#ifdef _WIN32
class external_mem_wrapper : public external_mem_wrapper_base {
public:
  /// Creates external memory resource wrapper for a given DX11 resource.
  /// \param [in] d3d11_res Pointer to the ID3D11Resource to be wrapped.
  /// \param [in] reg_flags Registration flags used for resource registration.
  external_mem_wrapper(ID3D11Resource *d3d11_res, unsigned reg_flags)
      : _res_D3D11(d3d11_res),
        external_mem_wrapper_base(
            sycl::ext::oneapi::experimental::external_mem(), reg_flags) {
    // Check if the resource is already registered to prevent double
    // registration.
    if (_res_D3D11_set.find(_res_D3D11) != _res_D3D11_set.end()) {
      throw std::runtime_error("This D3D11 resource is already registered!");
    }

    // Query the information from D3D11 resource object
    query_res_info(_res_D3D11);

    // Create a shared handle for the DX11 resource to enable interoperability.
    _res_win_nt_handle = create_shared_handle(_res_D3D11);

    // Prepare the external memory resource descriptor for SYCL interop.
    sycl::ext::oneapi::experimental::resource_win32_handle ext_mem_win_handle{
        _res_win_nt_handle};

    sycl::ext::oneapi::experimental::external_mem_descriptor<
        sycl::ext::oneapi::experimental::resource_win32_handle>
        ext_mem_desc{ext_mem_win_handle,
                     sycl::ext::oneapi::experimental::external_mem_handle_type::
                         win32_nt_dx12_resource,
                     _res_size_bytes};

    // Import resource's external memory into SYCL for use with SYCL APIs.
    _res_external_mem = sycl::ext::oneapi::experimental::external_mem(
        sycl::ext::oneapi::experimental::import_external_memory(
            ext_mem_desc, get_default_queue()));

    // Register the resource to track its uniqueness.
    _res_D3D11_set.insert(_res_D3D11);
  }

  /// Cleans up the external mem wrapper by releasing the SYCL external
  /// memory handle and the shared handle created for the DX11 resource.
  virtual ~external_mem_wrapper() {
    // Release the mutex when done
    _res_keyed_mutex->ReleaseSync(0);

    // Release the keyed mutex interface
    _res_keyed_mutex->Release();

    // Close the shared handle to the DX11 resource.
    CloseHandle(_res_win_nt_handle);

    // Release the resource's external memory associated with the SYCL external
    // mem handle.
    sycl::ext::oneapi::experimental::release_external_memory(
        get_external_mem(), get_default_queue());

    // Remove the resource from the tracking set.
    if (_res_D3D11_set.find(_res_D3D11) != _res_D3D11_set.end()) {
      _res_D3D11_set.erase(_res_D3D11);
    }
  }

private:
  inline static std::set<ID3D11Resource *> _res_D3D11_set;

  ID3D11Resource *_res_D3D11 = nullptr;
  IDXGIKeyedMutex *_res_keyed_mutex = nullptr;
  HANDLE _res_win_nt_handle = nullptr;

  /// Helper function to query the properties of DX11 resource
  void query_res_info(ID3D11Resource *resource) {
    unsigned int res_width = 0;
    unsigned int res_height = 0;
    unsigned int res_depth = 0;
    unsigned int res_num_levels = 1;
    unsigned int res_arr_size = 1;

    image_channel channel;

    // Get the dimension info of DX11 resource
    D3D11_RESOURCE_DIMENSION dimension;
    resource->GetType(&dimension);

    switch (dimension) {
    case D3D11_RESOURCE_DIMENSION_BUFFER: {
      _res_is_buffer = true;

      D3D11_BUFFER_DESC desc;
      ((ID3D11Buffer *)resource)->GetDesc(&desc);

      _res_size_bytes = desc.ByteWidth;
      break;
    }
    case D3D11_RESOURCE_DIMENSION_TEXTURE1D: {
      D3D11_TEXTURE1D_DESC desc;
      ((ID3D11Texture1D *)resource)->GetDesc(&desc);

      res_width = desc.Width;
      res_arr_size = desc.ArraySize;
      res_num_levels = (desc.MipLevels ? desc.MipLevels : 1);

      channel = get_img_ch_info(desc.Format);
      _res_size_bytes = res_width;
      break;
    }
    case D3D11_RESOURCE_DIMENSION_TEXTURE2D: {
      D3D11_TEXTURE2D_DESC desc;
      ((ID3D11Texture2D *)resource)->GetDesc(&desc);

      res_width = desc.Width;
      res_height = desc.Height;
      res_arr_size = desc.ArraySize;
      res_num_levels = (desc.MipLevels ? desc.MipLevels : 1);

      channel = get_img_ch_info(desc.Format);
      _res_size_bytes = res_width * res_height;
      break;
    }
    case D3D11_RESOURCE_DIMENSION_TEXTURE3D: {
      D3D11_TEXTURE3D_DESC desc;
      ((ID3D11Texture3D *)resource)->GetDesc(&desc);

      res_width = desc.Width;
      res_height = desc.Height;
      res_depth = desc.Depth;
      res_num_levels = (desc.MipLevels ? desc.MipLevels : 1);

      channel = get_img_ch_info(desc.Format);
      _res_size_bytes = res_width * res_height * res_depth;
      break;
    }
    default:
      throw std::runtime_error("Unsupported DX11 resource type!");
      break;
    }

    if (!_res_is_buffer) {
      if (res_arr_size > 1) {
        throw std::runtime_error(
            "Images with multiple array levels are not supported. "
            "Only standard and mipmap images are supported!");
      }

      // Calculate the size of a resource in bytes by multiplyting its dims
      // with arr size, num levels & channel size
      _res_size_bytes *=
          res_arr_size * res_num_levels * channel.get_total_size();

      sycl::ext::oneapi::experimental::image_type img_type =
          (res_num_levels > 1)
              ? sycl::ext::oneapi::experimental::image_type::mipmap
              : sycl::ext::oneapi::experimental::image_type::standard;

      _res_img_desc = sycl::ext::oneapi::experimental::image_descriptor{
          sycl::range<3>{res_width, res_height, res_depth},
          channel.get_channel_num(),
          channel.get_channel_type(),
          img_type,
          res_num_levels,
          res_arr_size};
    }
  }

  /// Helper function to create shared handle for DXD11 resource
  HANDLE create_shared_handle(ID3D11Resource *resource) {
    HANDLE win_nt_handle = nullptr;

    IDXGIResource1 *pDXGIResource1;
    resource->QueryInterface(__uuidof(IDXGIResource1),
                             reinterpret_cast<void **>(&pDXGIResource1));

    if (_res_is_buffer) {
      pDXGIResource1->GetSharedHandle(&win_nt_handle);
    } else {
      pDXGIResource1->CreateSharedHandle(
          nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
          nullptr, &win_nt_handle);
    }

    pDXGIResource1->Release();

    // Query for the keyed mutex interface
    resource->QueryInterface(__uuidof(IDXGIKeyedMutex),
                             reinterpret_cast<void **>(&_res_keyed_mutex));

    // Use the keyed mutex to synchronize access to the shared resource
    _res_keyed_mutex->AcquireSync(0, INFINITE);

    return win_nt_handle;
  }

  /// Helper function to get the texture channel info of DX11 resource using
  /// DXGI format
  image_channel get_img_ch_info(DXGI_FORMAT format) {
    image_channel channel;

    switch (format) {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
      channel = image_channel::create<sycl::half4>();
      break;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
      channel = image_channel::create<sycl::uchar4>();
      break;
    case DXGI_FORMAT_R16G16_FLOAT:
      channel = image_channel::create<sycl::half2>();
      break;
    case DXGI_FORMAT_R32_FLOAT:
      channel = image_channel::create<float>();
      break;
    default:
      throw std::runtime_error("Unsupported DX11 resource format!");
      break;
    }

    return channel;
  }
};
#endif // _WIN32

namespace detail {
struct sampled_image_handle_compare {
  template <class T> bool operator()(T L, T R) const {
    return L.raw_handle < R.raw_handle;
  }
};

inline std::pair<image_data, sampling_info> &get_img_info_map(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  static std::map<sycl::ext::oneapi::experimental::sampled_image_handle,
                  std::pair<image_data, sampling_info>,
                  sampled_image_handle_compare>
      img_info_map;
  return img_info_map[handle];
}

inline std::pair<image_data, sampling_info> &get_img_info_map(
    const sycl::ext::oneapi::experimental::unsampled_image_handle handle) {
  static std::map<sycl::ext::oneapi::experimental::unsampled_image_handle,
                  std::pair<image_data, sampling_info>,
                  sampled_image_handle_compare>
      img_info_map;
  return img_info_map[handle];
}

static inline size_t
get_ele_size(const sycl::ext::oneapi::experimental::image_descriptor &decs) {
  size_t channel_size;
  switch (decs.channel_type) {
  case sycl::image_channel_type::signed_int8:
  case sycl::image_channel_type::unsigned_int8:
    channel_size = 1;
    break;
  case sycl::image_channel_type::fp16:
  case sycl::image_channel_type::signed_int16:
  case sycl::image_channel_type::unsigned_int16:
    channel_size = 2;
    break;
  case sycl::image_channel_type::fp32:
  case sycl::image_channel_type::signed_int32:
  case sycl::image_channel_type::unsigned_int32:
    channel_size = 4;
    break;
  default:
    throw std::runtime_error("Unsupported channel_type in get_ele_size!");
    break;
  }
  return decs.num_channels * channel_size;
}

static inline sycl::event
dpct_memcpy(sycl::ext::oneapi::experimental::image_mem_handle src,
            const sycl::ext::oneapi::experimental::image_descriptor &desc_src,
            size_t w_offset_src, size_t h_offset_src, void *dest, size_t p,
            size_t w, size_t h, sycl::queue q) {
  const auto ele_size = get_ele_size(desc_src);
  const auto src_offset =
      sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
  const auto dest_offset = sycl::range<3>(0, 0, 0);
  const auto dest_extend = sycl::range<3>(p / ele_size, 0, 0);
  const auto copy_extend = sycl::range<3>(w / ele_size, h, 0);
  return q.ext_oneapi_copy(src, src_offset, desc_src, dest, dest_offset,
                           dest_extend, copy_extend);
}

static inline std::vector<sycl::event> dpct_memcpy_to_host(
    sycl::ext::oneapi::experimental::image_mem_handle src,
    const sycl::ext::oneapi::experimental::image_descriptor &desc_src,
    size_t w_offset_src, size_t h_offset_src, void *dest_host_ptr, size_t s,
    sycl::queue q) {
  std::vector<sycl::event> event_list;
  const auto ele_size = get_ele_size(desc_src);
  const auto w = desc_src.width * ele_size;
  size_t offset_dest = 0;
  while (s - offset_dest > w - w_offset_src) {
    const auto src_offset =
        sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
    const auto dest_offset = sycl::range<3>(offset_dest / ele_size, 0, 0);
    const auto dest_extend = sycl::range<3>(0, 0, 0);
    const auto copy_extend =
        sycl::range<3>((w - w_offset_src) / ele_size, 1, 0);
    event_list.push_back(q.ext_oneapi_copy(src, src_offset, desc_src,
                                           dest_host_ptr, dest_offset,
                                           dest_extend, copy_extend));
    offset_dest += w - w_offset_src;
    w_offset_src = 0;
    ++h_offset_src;
  }
  if (s - offset_dest > 0) {
    const auto src_offset =
        sycl::range<3>(w_offset_src / ele_size, h_offset_src, 0);
    const auto dest_offset = sycl::range<3>(offset_dest / ele_size, 0, 0);
    const auto dest_extend = sycl::range<3>(0, 0, 0);
    const auto copy_extend = sycl::range<3>((s - offset_dest) / ele_size, 1, 0);
    event_list.push_back(q.ext_oneapi_copy(src, src_offset, desc_src,
                                           dest_host_ptr, dest_offset,
                                           dest_extend, copy_extend));
  }
  return event_list;
}

static inline std::vector<sycl::event>
dpct_memcpy(sycl::ext::oneapi::experimental::image_mem_handle src,
            const sycl::ext::oneapi::experimental::image_descriptor &desc_src,
            size_t w_offset_src, size_t h_offset_src, void *dest, size_t s,
            sycl::queue q) {
  if (dpct::detail::get_pointer_attribute(q, dest) ==
      dpct::detail::pointer_access_attribute::device_only) {
    std::vector<sycl::event> event_list;
    dpct::detail::host_buffer buf(s, q, event_list);
    auto copy_events = dpct_memcpy_to_host(src, desc_src, w_offset_src,
                                           h_offset_src, buf.get_ptr(), s, q);
    event_list.push_back(dpct::detail::dpct_memcpy(
        q, dest, buf.get_ptr(), s, memcpy_direction::host_to_device,
        copy_events));
    return event_list;
  }
  return dpct_memcpy_to_host(src, desc_src, w_offset_src, h_offset_src, dest, s,
                             q);
}

static inline sycl::event
dpct_memcpy(const void *src,
            sycl::ext::oneapi::experimental::image_mem_handle dest,
            const sycl::ext::oneapi::experimental::image_descriptor &desc_dest,
            size_t w_offset_dest, size_t h_offset_dest, size_t p, size_t w,
            size_t h, sycl::queue q) {
  const auto ele_size = get_ele_size(desc_dest);
  const auto src_offset = sycl::range<3>(0, 0, 0);
  const auto src_extend = sycl::range<3>(p / ele_size, 0, 0);
  const auto dest_offset =
      sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
  const auto copy_extend = sycl::range<3>(w / ele_size, h, 0);
  // TODO: Remove const_cast after refining the signature of ext_oneapi_copy.
  return q.ext_oneapi_copy(const_cast<void *>(src), src_offset, src_extend,
                           dest, dest_offset, desc_dest, copy_extend);
}

static inline std::vector<sycl::event> dpct_memcpy_from_host(
    const void *src_host_ptr,
    sycl::ext::oneapi::experimental::image_mem_handle dest,
    const sycl::ext::oneapi::experimental::image_descriptor &desc_dest,
    size_t w_offset_dest, size_t h_offset_dest, size_t s, sycl::queue q) {
  std::vector<sycl::event> event_list;
  const auto ele_size = get_ele_size(desc_dest);
  const auto w = desc_dest.width * ele_size;
  size_t offset_src = 0;
  while (s - offset_src >= w - w_offset_dest) {
    const auto src_offset = sycl::range<3>(offset_src / ele_size, 0, 0);
    const auto src_extend = sycl::range<3>(0, 0, 0);
    const auto dest_offset =
        sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
    const auto copy_extend =
        sycl::range<3>((w - w_offset_dest) / ele_size, 1, 0);
    // TODO: Remove const_cast after refining the signature of ext_oneapi_copy.
    event_list.push_back(q.ext_oneapi_copy(
        const_cast<void *>(src_host_ptr), src_offset, src_extend, dest,
        dest_offset, desc_dest, copy_extend));
    offset_src += w - w_offset_dest;
    w_offset_dest = 0;
    ++h_offset_dest;
  }
  if (s - offset_src > 0) {
    const auto src_offset = sycl::range<3>(offset_src / ele_size, 0, 0);
    const auto src_extend = sycl::range<3>(0, 0, 0);
    const auto dest_offset =
        sycl::range<3>(w_offset_dest / ele_size, h_offset_dest, 0);
    const auto copy_extend = sycl::range<3>((s - offset_src) / ele_size, 1, 0);
    // TODO: Remove const_cast after refining the signature of ext_oneapi_copy.
    event_list.push_back(q.ext_oneapi_copy(
        const_cast<void *>(src_host_ptr), src_offset, src_extend, dest,
        dest_offset, desc_dest, copy_extend));
  }
  return event_list;
}

static inline std::vector<sycl::event> dpct_memcpy(
    const void *src, sycl::ext::oneapi::experimental::image_mem_handle dest,
    const sycl::ext::oneapi::experimental::image_descriptor &desc_dest,
    size_t w_offset_dest, size_t h_offset_dest, size_t s, sycl::queue q) {
  if (dpct::detail::get_pointer_attribute(q, src) ==
      dpct::detail::pointer_access_attribute::device_only) {
    std::vector<sycl::event> event_list;
    dpct::detail::host_buffer buf(s, q, event_list);
    event_list.push_back(dpct::detail::dpct_memcpy(
        q, buf.get_ptr(), src, s, memcpy_direction::device_to_host));
    auto copy_events = dpct_memcpy_from_host(
        buf.get_ptr(), dest, desc_dest, w_offset_dest, h_offset_dest, s, q);
    event_list.insert(event_list.end(), copy_events.begin(), copy_events.end());
    return event_list;
  }
  return dpct_memcpy_from_host(src, dest, desc_dest, w_offset_dest,
                               h_offset_dest, s, q);
}

static inline sycl::event
dpct_memcpy(const image_mem_wrapper *src, const sycl::id<3> &src_id,
            const size_t src_x_offest_byte, pitched_data &dest,
            const sycl::id<3> &dest_id, const size_t dest_x_offest_byte,
            const sycl::range<3> &size, const size_t copy_x_size_byte,
            sycl::queue q) {
  const auto ele_size = get_ele_size(src->get_desc());
  const auto src_offset = sycl::range<3>(
      src_x_offest_byte != 0 ? src_x_offest_byte / ele_size : src_id[0],
      src_id[1], src_id[2]);
  const auto dest_offset = sycl::range<3>(
      dest_x_offest_byte != 0 ? dest_x_offest_byte / ele_size : dest_id[0],
      dest_id[1], dest_id[2]);
  const auto dest_extend =
      sycl::range<3>(dest.get_pitch() / ele_size, dest.get_y(), 1);
  const auto copy_extend = sycl::range<3>(
      copy_x_size_byte != 0 ? copy_x_size_byte / ele_size : size[0], size[1],
      size[2]);
  return q.ext_oneapi_copy(src->get_handle(), src_offset, src->get_desc(),
                           dest.get_data_ptr(), dest_offset, dest_extend,
                           copy_extend);
}

static inline sycl::event
dpct_memcpy(pitched_data src, const sycl::id<3> &src_id,
            const size_t src_x_offest_byte, image_mem_wrapper *dest,
            const sycl::id<3> &dest_id, const size_t dest_x_offest_byte,
            const sycl::range<3> &size, const size_t copy_x_size_byte,
            sycl::queue q) {
  const auto ele_size = get_ele_size(dest->get_desc());
  const auto src_offset = sycl::range<3>(
      src_x_offest_byte != 0 ? src_x_offest_byte / ele_size : src_id[0],
      src_id[1], src_id[2]);
  const auto src_extend =
      sycl::range<3>(src.get_pitch() / ele_size, src.get_y(), 1);
  const auto dest_offset = sycl::range<3>(
      dest_x_offest_byte != 0 ? dest_x_offest_byte / ele_size : dest_id[0],
      dest_id[1], dest_id[2]);
  const auto copy_extend = sycl::range<3>(
      copy_x_size_byte != 0 ? copy_x_size_byte / ele_size : size[0], size[1],
      size[2]);
  return q.ext_oneapi_copy(src.get_data_ptr(), src_offset, src_extend,
                           dest->get_handle(), dest_offset, dest->get_desc(),
                           copy_extend);
}

static inline sycl::event
dpct_memcpy(const image_mem_wrapper *src, const sycl::id<3> &src_id,
            const size_t src_x_offest_byte, image_mem_wrapper *dest,
            const sycl::id<3> &dest_id, const size_t dest_x_offest_byte,
            const sycl::range<3> &size, const size_t copy_x_size_byte,
            sycl::queue q) {
  // TODO: Need change logic when sycl support image_mem to image_mem copy.
  const auto from_ele_size = get_ele_size(src->get_desc());
  const auto to_ele_size = get_ele_size(dest->get_desc());
  const auto copy_extend = sycl::range<3>(
      copy_x_size_byte != 0 ? copy_x_size_byte / from_ele_size : size[0],
      size[1], size[2]);
  std::vector<sycl::event> event_list;
  dpct::detail::host_buffer buf(
      copy_extend.size() * std::max(from_ele_size, to_ele_size), q, event_list);
  auto to = pitched_data(buf.get_ptr(), copy_extend[0] * from_ele_size,
                         copy_extend[0], copy_extend[1]);
  dpct_memcpy(src, src_id, src_x_offest_byte, to, sycl::id<3>(0, 0, 0), 0,
              copy_extend, 0, q);
  auto from = pitched_data(buf.get_ptr(), copy_extend[0] * to_ele_size,
                           copy_extend[0], copy_extend[1]);
  event_list.push_back(dpct_memcpy(from, sycl::id<3>(0, 0, 0), 0, dest, dest_id,
                                   dest_x_offest_byte, copy_extend, 0, q));
  return event_list.front();
}

template <typename T>
inline bool check_duplicate_entries(int count, T **entries) {
  std::set<T *> uniqueEntries;

  for (size_t i = 0; i < count; ++i) {
    auto result = uniqueEntries.insert(entries[i]);
    if (!result.second) {
      return true;
    }
  }

  return false;
}
} // namespace detail

/// Get image_mem_wrapper according to sampled image handle.
/// \param [in] handle The bindless image handle.
/// \returns The image_mem_wrapper of sampled image.
inline image_mem_wrapper *&get_img_mem(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  static std::map<sycl::ext::oneapi::experimental::sampled_image_handle,
                  image_mem_wrapper *, detail::sampled_image_handle_compare>
      img_mem_map;
  return img_mem_map[handle];
}

/// Get image_mem_wrapper according to unsampled image handle.
/// \param [in] handle The unsampled bindless image handle.
/// \returns The image_mem_wrapper of unsampled image.
inline image_mem_wrapper *&get_img_mem(
    const sycl::ext::oneapi::experimental::unsampled_image_handle handle) {
  static std::map<sycl::ext::oneapi::experimental::unsampled_image_handle,
                  image_mem_wrapper *, detail::sampled_image_handle_compare>
      img_mem_map;
  return img_mem_map[handle];
}

/// Associate an image memory wrapper with a bindless unsampled image handle in
/// the global registry.
///
/// Inserts or updates an entry in the internal image memory map, using the
/// specified bindless unsampled image handle as the key and the provided image
/// memory wrapper as the mapped value.
/// \param [in] handle The unsampled bindless image handle.
/// \param [in] img_mem The image_mem_wrapper associated with the unsampled handle.
static inline void set_img_mem(
    const sycl::ext::oneapi::experimental::unsampled_image_handle handle,
    image_mem_wrapper *img_mem) {
  dpct::experimental::get_img_mem(handle) = img_mem;
}

/// Associate an image memory wrapper with a bindless sampled image handle in
/// the global registry.
///
/// Inserts or updates an entry in the internal image memory map, using the
/// specified bindless unsampled image handle as the key and the provided image
/// memory wrapper as the mapped value.
/// \param [in] handle The sampled bindless image handle.
/// \param [in] img_mem The image_mem_wrapper associated with the sampled handle.
static inline void set_img_mem(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle,
    image_mem_wrapper *img_mem) {
  dpct::experimental::get_img_mem(handle) = img_mem;
}

#ifdef _WIN32
/// Map the resource memories to mem handles
/// \param [in] count The count of resources to map.
/// \param [in] handles The external mem wrappers used to map the resources.
/// \param [in] q The queue used to map the resource with.
inline void map_resources(int count, external_mem_wrapper **handles,
                          queue_ptr q_ptr = &get_default_queue()) {
  if (detail::check_duplicate_entries(count, handles)) {
    throw std::runtime_error(
        "Duplicate handle entries found during resource mapping!");
  }

  for (int i = 0; i < count; i++) {
    handles[i]->map_resource(*q_ptr);
  }
}

/// Unmap the resource memories to mem handles
/// \param [in] count The count of resources to unmap.
/// \param [in] handles The external mem wrappers used to unmap the resources.
/// \param [in] q The queue used to unmap the resource with.
inline void unmap_resources(int count, external_mem_wrapper **handles,
                            queue_ptr q_ptr = &get_default_queue()) {
  if (detail::check_duplicate_entries(count, handles)) {
    throw std::runtime_error(
        "Duplicate handle entries found during resource unmapping!");
  }

  for (int i = 0; i < count; i++) {
    handles[i]->unmap_resource(*q_ptr);
  }
}
#endif // _WIN32

/// Imports resource's external memory into a SYCL external memory object.
/// \param [in] extMem SYCL external memory object ptr to be initialized.
/// \param [in] memHandleDesc Pointer to the external memory resource handle
/// descriptor.
/// \throws std::runtime_error If no valid external memory resource handle is
/// provided.
inline void
import_external_memory(sycl::ext::oneapi::experimental::external_mem *extMem,
                       external_mem_handle_desc *memHandleDesc) {
#ifdef _WIN32
  if (memHandleDesc->get_win32_handle()) {
    *extMem = sycl::ext::oneapi::experimental::import_external_memory(
        sycl::ext::oneapi::experimental::external_mem_descriptor<
            sycl::ext::oneapi::experimental::resource_win32_handle>{
            {memHandleDesc->get_win32_handle()},
            memHandleDesc->get_handle_type(),
            memHandleDesc->get_res_size()},
        get_default_queue());
  } else if (memHandleDesc->get_win32_obj_name()) {
    throw std::runtime_error(
        "Importing external memory resource using Windows NT name"
        " type is not supported. Only NT handle type is"
        " supported");
  }
#else
  if (memHandleDesc->get_fd_handle() != -1) {
    *extMem = sycl::ext::oneapi::experimental::import_external_memory(
        sycl::ext::oneapi::experimental::external_mem_descriptor<
            sycl::ext::oneapi::experimental::resource_fd>{
            {memHandleDesc->get_fd_handle()},
            memHandleDesc->get_handle_type(),
            memHandleDesc->get_res_size()},
        get_default_queue());
  }
#endif // _WIN32
  else {
    throw std::runtime_error(
        "No external memory resource handle is set for importing!");
  }
}

/// Imports external semaphore resource into a SYCL external semaphore object.
/// \param [in] extSem SYCL external semaphore object ptr to be initialized.
/// \param [in] semHandleDesc Pointer to the external semaphore resource handle
/// descriptor.
/// \throws std::runtime_error If no valid external semaphore resource handle is
/// provided.
inline void import_external_semaphore(external_sem_wrapper **extSem,
                                      external_sem_handle_desc *semHandleDesc) {
  sycl::ext::oneapi::experimental::external_semaphore extSemObj;
#ifdef _WIN32
  if (semHandleDesc->get_win32_handle()) {
    extSemObj = sycl::ext::oneapi::experimental::import_external_semaphore(
        sycl::ext::oneapi::experimental::external_semaphore_descriptor<
            sycl::ext::oneapi::experimental::resource_win32_handle>{
            {semHandleDesc->get_win32_handle()},
            semHandleDesc->get_handle_type()},
        get_default_queue());
  } else if (semHandleDesc->get_win32_obj_name()) {
    throw std::runtime_error(
        "Importing external semaphore resource using Windows NT name"
        " type is not supported. Only NT handle type is"
        " supported");
  }
#else
  if (semHandleDesc->get_fd_handle() != -1) {
    extSemObj = sycl::ext::oneapi::experimental::import_external_semaphore(
        sycl::ext::oneapi::experimental::external_semaphore_descriptor<
            sycl::ext::oneapi::experimental::resource_fd>{
            {semHandleDesc->get_fd_handle()}, semHandleDesc->get_handle_type()},
        get_default_queue());
  }
#endif // _WIN32
  else {
    throw std::runtime_error(
        "No external semaphore resource handle is set for importing!");
  }

  *extSem =
      new external_sem_wrapper(extSemObj, semHandleDesc->get_handle_type());
}

/// Signals the external semaphore resource.
/// \param [in] extSem ptr to dpct external semaphore resource ptr
/// \param [in] semSignalParams Pointer to the external semaphore signal
/// parameters.
/// \param [in] numExtSems Number of external semaphores to signal.
/// \param [in] q_ptr The queue used to signal the external semaphore resource.
inline void signal_external_semaphores(external_sem_wrapper **extSem,
                                       external_sem_params *semSignalParams,
                                       unsigned int numExtSems,
                                       queue_ptr q_ptr = &get_default_queue()) {
  for (int i = 0; i < numExtSems; i++) {
    auto extSemType = extSem[i]->get_handle_type();
    switch (extSemType) {
#ifdef _WIN32
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        win32_nt_dx12_fence:
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        timeline_win32_nt_handle:
      q_ptr->ext_oneapi_signal_external_semaphore(
          extSem[i]->get(), semSignalParams[i].get_value());
      break;
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        win32_nt_handle:
      q_ptr->ext_oneapi_signal_external_semaphore(extSem[i]->get());
      break;
#else  // _WIN32
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        timeline_fd:
      q_ptr->ext_oneapi_signal_external_semaphore(
          extSem[i]->get(), semSignalParams[i].get_value());
      break;
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        opaque_fd:
      q_ptr->ext_oneapi_signal_external_semaphore(extSem[i]->get());
      break;
#endif // _WIN32
    default:
      throw std::runtime_error(
          "Unsupported external semaphore resource handle type!");
      break;
    }
  }
}

/// Wait for the external semaphore resource.
/// \param [in] extSem ptr to dpct external semaphore resource ptr
/// \param [in] semWaitParams Pointer to the external semaphore wait parameters.
/// \param [in] numExtSems Number of external semaphores to wait.
/// \param [in] q_ptr The queue used to wait on the external semaphore resource.
inline void wait_external_semaphores(external_sem_wrapper **extSem,
                                     external_sem_params *semWaitParams,
                                     unsigned int numExtSems,
                                     queue_ptr q_ptr = &get_default_queue()) {
  for (int i = 0; i < numExtSems; i++) {
    auto extSemType = extSem[i]->get_handle_type();
    switch (extSemType) {
#ifdef _WIN32
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        win32_nt_dx12_fence:
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        timeline_win32_nt_handle:
      q_ptr->ext_oneapi_wait_external_semaphore(extSem[i]->get(),
                                                semWaitParams[i].get_value());
      break;
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        win32_nt_handle:
      q_ptr->ext_oneapi_wait_external_semaphore(extSem[i]->get());
      break;
#else  // _WIN32
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        timeline_fd:
      q_ptr->ext_oneapi_wait_external_semaphore(extSem[i]->get(),
                                                semWaitParams[i].get_value());
      break;
    case sycl::ext::oneapi::experimental::external_semaphore_handle_type::
        opaque_fd:
      q_ptr->ext_oneapi_wait_external_semaphore(extSem[i]->get());
      break;
#endif // _WIN32
    default:
      throw std::runtime_error(
          "Unsupported external semaphore resource handle type!");
      break;
    }
  }
}

/// Create bindless image according to image data and sampling info.
/// \param [in] data The image data used to create bindless image.
/// \param [in] info The image sampling info used to create bindless image.
/// \param [in] q The queue where the image creation be executed.
/// \returns The sampled image handle of created bindless image.
static inline sycl::ext::oneapi::experimental::sampled_image_handle
create_bindless_image(image_data data, sampling_info info,
                      sycl::queue q = get_default_queue()) {
  auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
      info.get_addressing_mode(), info.get_coordinate_normalization_mode(),
      info.get_filtering_mode(), info.get_mipmap_filtering(),
      info.get_min_mipmap_level_clamp(), info.get_max_mipmap_level_clamp(),
      info.get_max_anisotropy());

  switch (data.get_data_type()) {
  case image_data_type::linear: {
    // linear memory only use sycl::filtering_mode::nearest.
    samp.filtering = sycl::filtering_mode::nearest;
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new image_mem_wrapper(
        data.get_channel(), data.get_x() / data.get_channel().get_total_size());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    set_img_mem(img, mem);
    auto ptr = data.get_data_ptr();
#ifdef DPCT_USM_LEVEL_NONE
    q.ext_oneapi_copy(get_buffer(ptr).get_host_access().get_pointer(),
                      mem->get_handle(), mem->get_desc())
        .wait();
#else
    q.ext_oneapi_copy(ptr, mem->get_handle(), mem->get_desc()).wait();
#endif
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  case image_data_type::pitch: {
#ifdef DPCT_USM_LEVEL_NONE
    auto mem =
        new image_mem_wrapper(data.get_channel(), data.get_x(), data.get_y());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    set_img_mem(img, mem);
    q.ext_oneapi_copy(
         get_buffer(data.get_data_ptr()).get_host_access().get_pointer(),
         mem->get_handle(), mem->get_desc())
        .wait();
#else
    auto desc = sycl::ext::oneapi::experimental::image_descriptor(
        {data.get_x(), data.get_y()}, data.get_channel().get_channel_num(),
        data.get_channel_type());
    auto img = sycl::ext::oneapi::experimental::create_image(
        data.get_data_ptr(), data.get_pitch(), samp, desc, q);
#endif
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  case image_data_type::matrix: {
    const auto mem = static_cast<image_mem_wrapper *>(data.get_data_ptr());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    detail::get_img_info_map(img) = {data, info};
    return img;
  }
  default:
    throw std::runtime_error(
        "Unsupported image_data_type in create_bindless_image!");
    break;
  }
  // Must not reach here.
  return sycl::ext::oneapi::experimental::sampled_image_handle();
}

/// Create bindless image according to image data.
/// \param [in] data The image data used to create bindless image.
/// \param [in] q The queue where the image creation be executed.
/// \returns The sampled image handle of created bindless image.
static inline sycl::ext::oneapi::experimental::unsampled_image_handle
create_bindless_image(image_data data, sycl::queue q = get_default_queue()) {
  switch (data.get_data_type()) {
  case image_data_type::linear: {
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new image_mem_wrapper(
        data.get_channel(), data.get_x() / data.get_channel().get_total_size());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), mem->get_desc(), q);
    set_img_mem(img, mem);
    auto ptr = data.get_data_ptr();
#ifdef DPCT_USM_LEVEL_NONE
    q.ext_oneapi_copy(get_buffer(ptr).get_host_access().get_pointer(),
                      mem->get_handle(), mem->get_desc())
        .wait();
#else
    q.ext_oneapi_copy(ptr, mem->get_handle(), mem->get_desc()).wait();
#endif
    return img;
  }
  case image_data_type::pitch: {
    auto mem =
        new image_mem_wrapper(data.get_channel(), data.get_x(), data.get_y());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), mem->get_desc(), q);
    set_img_mem(img, mem);
#ifdef DPCT_USM_LEVEL_NONE
    q.ext_oneapi_copy(
         get_buffer(data.get_data_ptr()).get_host_access().get_pointer(),
         mem->get_handle(), mem->get_desc())
        .wait();
#else
    q.ext_oneapi_copy(data.get_data_ptr(), mem->get_handle(), mem->get_desc())
        .wait();

#endif
    return img;
  }
  case image_data_type::matrix: {
    const auto mem = static_cast<image_mem_wrapper *>(data.get_data_ptr());
    auto img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), mem->get_desc(), q);
    return img;
  }
  default:
    throw std::runtime_error(
        "Unsupported image_data_type in create_bindless_image!");
    break;
  }
  // Must not reach here.
  return sycl::ext::oneapi::experimental::unsampled_image_handle();
}

/// Destroy bindless image.
/// \param [in] handle The bindless image should be destroyed.
/// \param [in] q The queue where the image destruction be executed.
template <class T>
static inline void destroy_bindless_image(T handle,
                                          sycl::queue q = get_default_queue()) {
  auto &mem = get_img_mem(handle);
  if (mem) {
    delete mem;
    mem = nullptr;
  }
  sycl::ext::oneapi::experimental::destroy_image_handle(handle, q);
}

/// Get the image data according to sampled image handle.
/// \param [in] handle The bindless image handle.
/// \returns The image data of sampled image.
template <class T> static inline image_data get_data(const T handle) {
  return detail::get_img_info_map(handle).first;
}

/// Get the sampling info according to sampled image handle.
/// \param [in] handle The bindless image handle.
/// \returns The sampling info of sampled image.
static inline sampling_info get_sampling_info(
    const sycl::ext::oneapi::experimental::sampled_image_handle handle) {
  return detail::get_img_info_map(handle).second;
}

/// The base class of different template specialization bindless_image_wrapper
/// class.
class bindless_image_wrapper_base {
public:
  bindless_image_wrapper_base() {
    // Make sure that singleton class dev_mgr will destruct later than this.
    dev_mgr::instance();
  }

  /// Destroy bindless image wrapper.
  ~bindless_image_wrapper_base() {
    destroy_bindless_image(_img, get_default_queue());
  }

  /// Attach linear data to bindless image.
  /// \param [in] data The linear data used to create bindless image.
  /// \param [in] size The size of linear data used to create bindless image.
  /// \param [in] channel The image channel used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t size, const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    // TODO: Use pointer to create image when bindless image support.
    auto mem = new image_mem_wrapper(channel, size);
    _img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    set_img_mem(_img, mem);
    auto ptr = data;
#ifdef DPCT_USM_LEVEL_NONE
    q.ext_oneapi_copy(get_buffer(data).get_host_access().get_pointer(),
                      mem->get_handle(), mem->get_desc())
        .wait();
#else
    q.ext_oneapi_copy(ptr, mem->get_handle(), mem->get_desc()).wait();
#endif
  }

  /// Attach linear data to bindless image.
  /// \param [in] data The linear data used to create bindless image.
  /// \param [in] size The size of linear data used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t size, sycl::queue q = get_default_queue()) {
    attach(data, size, _channel, q);
  }

  /// Attach device_ptr data to bindless image.
  /// \param [in] desc The image_descriptor used to create bindless image.
  /// \param [in] ptr The data pointer used to create bindless image.
  /// \param [in] pitch The pitch of 2D data used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(const sycl::ext::oneapi::experimental::image_descriptor *desc,
              device_ptr ptr, size_t pitch,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
#ifdef DPCT_USM_LEVEL_NONE
    auto mem = new image_mem_wrapper(desc);
    _img = sycl::ext::oneapi::experimental::create_image(mem->get_handle(),
                                                         samp, *desc, q);
    set_img_mem(_img, mem);
    q.ext_oneapi_copy(get_buffer(ptr).get_host_access().get_pointer(),
                      mem->get_handle(), mem->get_desc())
        .wait();
#else
    _img = sycl::ext::oneapi::experimental::create_image(ptr, pitch, samp,
                                                         *desc, q);
#endif
  }

  /// Attach 2D data to bindless image.
  /// \param [in] data The 2D data used to create bindless image.
  /// \param [in] width The width of 2D data used to create bindless image.
  /// \param [in] height The height of 2D data used to create bindless image.
  /// \param [in] pitch The pitch of 2D data used to create bindless image.
  /// \param [in] channel The image channel used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t width, size_t height, size_t pitch,
              const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    auto desc = sycl::ext::oneapi::experimental::image_descriptor(
        {width, height}, channel.get_channel_num(), channel.get_channel_type());
    attach(&desc, static_cast<device_ptr>(data), pitch, q);
  }

  /// Attach 2D data to bindless image.
  /// \param [in] data The 2D data used to create bindless image.
  /// \param [in] width The width of 2D data used to create bindless image.
  /// \param [in] height The height of 2D data used to create bindless image.
  /// \param [in] pitch The pitch of 2D data used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(void *data, size_t width, size_t height, size_t pitch,
              sycl::queue q = get_default_queue()) {
    attach(data, width, height, pitch, _channel, q);
  }

  /// Attach image memory to bindless image.
  /// \param [in] mem The image memory used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(image_mem_wrapper *mem, const image_channel &channel,
              sycl::queue q = get_default_queue()) {
    detach(q);
    auto samp = sycl::ext::oneapi::experimental::bindless_image_sampler(
        _addressing_mode, _coordinate_normalization_mode, _filtering_mode);
    _img = sycl::ext::oneapi::experimental::create_image(
        mem->get_handle(), samp, mem->get_desc(), q);
    set_img_mem(_img, mem);
  }

  /// Attach image memory to bindless image.
  /// \param [in] mem The image memory used to create bindless image.
  /// \param [in] q The queue where the image creation be executed.
  void attach(image_mem_wrapper *mem, sycl::queue q = get_default_queue()) {
    attach(mem, _channel, q);
  }

  /// Detach bindless image data.
  /// \param [in] q The queue where the image destruction be executed.
  void detach(sycl::queue q = get_default_queue()) {
    destroy_bindless_image(_img, q);
  }

  /// Set image channel of bindless image.
  /// \param [in] channel The image channel used to set.
  void set_channel(image_channel channel) { _channel = channel; }
  /// Get image channel of bindless image.
  /// \returns The image channel of bindless image.
  image_channel get_channel() { return _channel; }

  /// Set channel size of image channel.
  /// \param [in] channel_num The channels number to set.
  /// \param [in] channel_size The size for each channel in bits.
  void set_channel_size(unsigned channel_num, unsigned channel_size) {
    return _channel.set_channel_size(channel_num, channel_size);
  }
  /// Get channel size of image channel.
  /// \returns The size for each channel in bits.
  unsigned get_channel_size() { return _channel.get_channel_size(); }

  /// Set image channel data type of image channel.
  /// \param [in] type The image channel data type to set.
  void set_channel_data_type(image_channel_data_type type) {
    _channel.set_channel_data_type(type);
  }
  /// Get image channel data type of image channel.
  /// \returns The image channel data type of image channel.
  image_channel_data_type get_channel_data_type() {
    return _channel.get_channel_data_type();
  }

  /// Set channel num of bindless image.
  /// \param [in] filtering_mode The channel num used to set.
  void set_channel_num(unsigned num) { return _channel.set_channel_num(num); }

  /// Set channel type of bindless image.
  /// \param [in] filtering_mode The channel type used to set.
  void set_channel_type(sycl::image_channel_type type) {
    return _channel.set_channel_type(type);
  }

  /// Set addressing mode of bindless image.
  /// \param [in] addressing_mode The addressing mode used to set.
  void set(sycl::addressing_mode addressing_mode) {
    _addressing_mode = addressing_mode;
  }
  /// Get addressing mode of bindless image.
  /// \returns The addressing mode of bindless image.
  sycl::addressing_mode get_addressing_mode() { return _addressing_mode; }

  /// Set coordinate normalization mode of bindless image.
  /// \param [in] coordinate_normalization_mode The coordinate normalization
  /// mode used to set.
  void set(sycl::coordinate_normalization_mode coordinate_normalization_mode) {
    _coordinate_normalization_mode = coordinate_normalization_mode;
  }
  /// Set coordinate normalization mode of bindless image.
  /// \param [in] is_normalized Determine whether the coordinate should be
  /// normalized.
  void set_coordinate_normalization_mode(int is_normalized) {
    _coordinate_normalization_mode =
        is_normalized ? sycl::coordinate_normalization_mode::normalized
                      : sycl::coordinate_normalization_mode::unnormalized;
  }
  /// Get coordinate normalization mode of bindless image.
  /// \returns The coordinate normalization mode of bindless image.
  bool is_coordinate_normalized() {
    return _coordinate_normalization_mode ==
           sycl::coordinate_normalization_mode::normalized;
  }

  /// Set filtering mode of bindless image.
  /// \param [in] filtering_mode The filtering mode used to set.
  void set(sycl::filtering_mode filtering_mode) {
    _filtering_mode = filtering_mode;
  }
  /// Get filtering mode of bindless image.
  /// \returns The filtering mode of bindless image.
  sycl::filtering_mode get_filtering_mode() { return _filtering_mode; }

  /// Get bindless image handle.
  /// \returns The sampled image handle of created bindless image.
  inline sycl::ext::oneapi::experimental::sampled_image_handle get_handle() {
    return _img;
  }

  /// Set mipmap sample filtering mode for bindless image handle
  /// \param [in] filtering_mode The mipmap filtering mode
  inline void set_mip_filtering_mode(sycl::filtering_mode mode) {
    auto sampling_info = get_sampling_info(_img);
    sampling_info.set_mipmap_filtering(mode);
  }

  /// Get mipmap sample filtering mode for bindless image handle
  /// \return The mipmap filtering mode
  inline sycl::filtering_mode get_mip_filtering_mode(void) {
    auto sampling_info = get_sampling_info(_img);
    return sampling_info.get_mipmap_filtering();
  }

  /// Get mipmap sample filtering mode for bindless image handle
  /// \return The mipmap filtering mode
  inline void get_mip_level_clamp(float *min_level_clamp,
                                  float *max_level_clamp) {
    auto sampling_info = get_sampling_info(_img);
    *min_level_clamp = sampling_info.get_min_mipmap_level_clamp();
    *max_level_clamp = sampling_info.get_max_mipmap_level_clamp();
  }

  /// Get mipmap memory wrapper attached the bindless image
  /// \return The mipmap memory wrapper
  inline image_mem_wrapper *get_attached_mipmap_data(void) {
    auto mem = get_img_mem(_img);

    if (mem->get_image_type() !=
        sycl::ext::oneapi::experimental::image_type::mipmap)
      throw std::runtime_error("Bindless image data is not of mipmap type");

    return mem;
  }

private:
  image_channel _channel;
  sycl::addressing_mode _addressing_mode = sycl::addressing_mode::clamp_to_edge;
  sycl::coordinate_normalization_mode _coordinate_normalization_mode =
      sycl::coordinate_normalization_mode::unnormalized;
  sycl::filtering_mode _filtering_mode = sycl::filtering_mode::nearest;
  sycl::ext::oneapi::experimental::sampled_image_handle _img{0};
};

/// The wrapper class of bindless image.
template <class T, int dimensions>
class bindless_image_wrapper : public bindless_image_wrapper_base {
public:
  /// Create bindless image wrapper according to template argument \p T.
  bindless_image_wrapper() { set_channel(image_channel::create<T>()); }
};

/// Asynchronously copies from the image memory to memory specified by a
/// pointer, The return of the function does NOT guarantee the copy is
/// completed.
/// \param [in] dest The destination memory address.
/// \param [in] p The pitch of destination memory.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void async_dpct_memcpy(void *dest, size_t p,
                                     const image_mem_wrapper *src,
                                     size_t w_offset_src, size_t h_offset_src,
                                     size_t w, size_t h,
                                     sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src->get_handle(), src->get_desc(), w_offset_src,
                      h_offset_src, dest, p, w, h, q);
}

/// Synchronously copies from the image memory to memory specified by a
/// pointer, The function will return after the copy is completed.
/// \param [in] dest The destination memory address.
/// \param [in] p The pitch of destination memory.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(void *dest, size_t p,
                               const image_mem_wrapper *src,
                               size_t w_offset_src, size_t h_offset_src,
                               size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src->get_handle(), src->get_desc(), w_offset_src,
                      h_offset_src, dest, p, w, h, q)
      .wait();
}

/// Asynchronously copies from the image memory to memory specified by a
/// pointer, The return of the function does NOT guarantee the copy is
/// completed.
/// \param [in] dest The destination memory address.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void async_dpct_memcpy(void *dest, const image_mem_wrapper *src,
                                     size_t w_offset_src, size_t h_offset_src,
                                     size_t s,
                                     sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src->get_handle(), src->get_desc(), w_offset_src,
                      h_offset_src, dest, s, q);
}

/// Synchronously copies from the image memory to memory specified by a
/// pointer, The function will return after the copy is completed.
/// \param [in] dest The destination memory address.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(void *dest, const image_mem_wrapper *src,
                               size_t w_offset_src, size_t h_offset_src,
                               size_t s, sycl::queue q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memcpy(src->get_handle(), src->get_desc(),
                                        w_offset_src, h_offset_src, dest, s,
                                        q));
}

/// Asynchronously copies from memory specified by a pointer to the image
/// memory, The return of the function does NOT guarantee the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source memory address.
/// \param [in] p The pitch of source memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void async_dpct_memcpy(image_mem_wrapper *dest,
                                     size_t w_offset_dest, size_t h_offset_dest,
                                     const void *src, size_t p, size_t w,
                                     size_t h,
                                     sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest->get_handle(), dest->get_desc(), w_offset_dest,
                      h_offset_dest, p, w, h, q);
}

/// Synchronously copies from memory specified by a pointer to the image
/// memory, The function will return after the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source memory address.
/// \param [in] p The pitch of source memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(image_mem_wrapper *dest, size_t w_offset_dest,
                               size_t h_offset_dest, const void *src, size_t p,
                               size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest->get_handle(), dest->get_desc(), w_offset_dest,
                      h_offset_dest, p, w, h, q)
      .wait();
}

/// Asynchronously copies from memory specified by a pointer to the image
/// memory, The return of the function does NOT guarantee the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source memory address.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void async_dpct_memcpy(image_mem_wrapper *dest,
                                     size_t w_offset_dest, size_t h_offset_dest,
                                     const void *src, size_t s,
                                     sycl::queue q = get_default_queue()) {
  detail::dpct_memcpy(src, dest->get_handle(), dest->get_desc(), w_offset_dest,
                      h_offset_dest, s, q);
}

/// Synchronously copies from memory specified by a pointer to the image
/// memory, The function will return after the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source memory address.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(image_mem_wrapper *dest, size_t w_offset_dest,
                               size_t h_offset_dest, const void *src, size_t s,
                               sycl::queue q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memcpy(src, dest->get_handle(),
                                        dest->get_desc(), w_offset_dest,
                                        h_offset_dest, s, q));
}

/// Synchronously copies from image memory to the image memory, The function
/// will return after the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] w The width of matrix to be copied.
/// \param [in] h The height of matrix to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(image_mem_wrapper *dest, size_t w_offset_dest,
                               size_t h_offset_dest,
                               const image_mem_wrapper *src,
                               size_t w_offset_src, size_t h_offset_src,
                               size_t w, size_t h,
                               sycl::queue q = get_default_queue()) {
  auto temp = (void *)sycl::malloc_device(w * h, q);
  // TODO: Need change logic when sycl support image_mem to image_mem copy.
  dpct_memcpy(temp, w, src, w_offset_src, h_offset_src, w, h, q);
  dpct_memcpy(dest, w_offset_dest, h_offset_dest, temp, w, w, h, q);
  sycl::free(temp, q);
}

/// Synchronously copies from image memory to the image memory, The function
/// will return after the copy is completed.
/// \param [in] dest The destination image memory.
/// \param [in] w_offset_dest The x offset of destination image memory.
/// \param [in] h_offset_dest The y offset of destination image memory.
/// \param [in] src The source image memory.
/// \param [in] w_offset_src The x offset of source image memory.
/// \param [in] h_offset_src The y offset of source image memory.
/// \param [in] s The size to be copied.
/// \param [in] q The queue to execute the copy task.
static inline void dpct_memcpy(image_mem_wrapper *dest, size_t w_offset_dest,
                               size_t h_offset_dest,
                               const image_mem_wrapper *src,
                               size_t w_offset_src, size_t h_offset_src,
                               size_t s, sycl::queue q = get_default_queue()) {
  auto temp = (void *)sycl::malloc_device(s, q);
  // TODO: Need change logic when sycl support image_mem to image_mem copy.
  dpct_memcpy(temp, src, w_offset_src, h_offset_src, s, q);
  dpct_memcpy(dest, w_offset_dest, h_offset_dest, temp, s, q);
  sycl::free(temp, q);
}
// A wrapper for sycl fetch_image function for the byte addressing image.
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT fetch_image_by_byte(
    const sycl::ext::oneapi::experimental::unsampled_image_handle &imageHandle,
    CoordT &&coords) {
  if constexpr (std::is_scalar_v<CoordT>) {
    return sycl::ext::oneapi::experimental::fetch_image<DataT, HintT, CoordT>(
        imageHandle, coords / sizeof(DataT));
  } else {
    coords[0] = coords[0] / sizeof(DataT);
    return sycl::ext::oneapi::experimental::fetch_image<DataT, HintT, CoordT>(
        imageHandle, coords);
  }
}

using image_mem_wrapper_ptr = image_mem_wrapper *;
#ifdef _WIN32
using external_mem_wrapper_ptr = external_mem_wrapper *;
#else
using external_mem_wrapper_ptr = external_mem_wrapper_base *;
#endif // _WIN32
using external_sem_wrapper_ptr = external_sem_wrapper *;
using bindless_image_wrapper_base_p = bindless_image_wrapper_base *;

#endif

} // namespace experimental
} // namespace dpct

#endif // !__DPCT_BINDLESS_IMAGE_HPP__
