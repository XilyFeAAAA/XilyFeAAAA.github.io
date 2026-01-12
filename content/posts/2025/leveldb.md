---
title: "LevelDB 设计与实现"
date: '2025-10-19T21:07:11+08:00'
authors: [Xilyfe]
tags: ["数据库", "NoSQL"]
---


![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1737352094026-0f819518-aaac-4acf-b912-e4f8e6d3e66c.png)

# 环境配置
### 安装 LevelDB 库
> github 上下载 LevelDB 源码
>

```plain
git clone https://github.com/google/leveldb.git
cd leveldb/third_party
git clone https://github.com/google/googletest.git
git clone https://github.com/google/benchmark.git
```

> visual studio 终端编译静态库
>

1. 打开终端

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736313994680-9387789b-5e52-4df7-be03-000bf3f8c0b0.png) 

2. 切换到项目路径下，cmake 编译

```cpp
cd Library\leveldb
cmake -B build -DCMAKE_INSTALL_PREFIX=安装路径
cmake --build build -j8 --target install
```

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736315909380-06634830-112a-40dc-9103-828377e9ce10.png)

3. 配置 visual studio

> 在解决方案中点击属性-链接器-输入-附加依赖项
>

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736316292140-105fbb86-9154-420e-b13e-fbfa7d9e67e3.png)

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1745738434091-2dcb0523-fc9e-4d2a-9c25-286be43226d8.png)

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1745835136419-38efc1fb-ff35-42e6-affd-8842780e2ddb.png)

> 添加 lib 文件夹中的 dll
>

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736316375195-56d14465-0ef4-462c-9093-6d975b32beb7.png)

> 点击 C/C++-常规-附加包含目录，加入 include 文件夹
>

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736316473992-d2f5c964-9e9e-4922-9e3e-48a85c7b891c.png)

# LevelDB 接口
## 实例化
> 由于对 LevelDB 操作需要实例化一个指针，所以采用 c++的智能指针，避免退出时没有 delete 
>

```cpp
#include<memeory>
leveldb::DB* dbptr = nullptr
leveldb::Options opts;
leveldb::Status status = leveldb::DB::Open(opts, file_path, &dbptr);
if (status.ok()){}
else{}
std::unique_ptr<leveldb::DB> db(dbptr);
```

> leveldb::Options 是用于配置数据库的一个结构体
>

```cpp
struct LEVELDB_EXPORT Options{
    const Comparator* comparator; // 比较器
    bool create_if_missing = false; // 如果数据库不存在，是否创建新数据库
    bool error_if_exists = false; // 如果数据库已存在，是否返回错误
    bool paranoid_checks = false;
    size_t write_buffer_size = 4 * 1024 * 1024; // 写缓冲区的大小。在写入数据时，LevelDB首先将数据放入写缓冲区（memtable）。当缓冲区填满后，数据会被刷写到磁盘
    int max_open_files = 1000; // 可以同时打开的文件的最大数量
    size_t block_size = 4 * 1024; // 每个块的大小。LevelDB把数据存储在按块组织的文件中，块是数据存储和索引的最小单元
    Cache* block_cache = nullptr; // BlockCache指针，默认8MB
};
```

## 写操作
### 点写
```cpp
leveldb::WriteOptions write_opts;
write_opts.sync = false; // default option
// 参数类型是slice，但是slice构造函数支持const char*，可以用隐式转换
db->Put(write_opts, "keyword", "value"); 
```

> leveldb::WriteOptions 是用于配置写操作的结构体
>

```cpp
struct LEVELDB_EXPORT WriteOptions {
    bool sync = false;
    // leveldb在写操作时是将数据写到cache中，没有真正写到磁盘里。如果sync置为false，那么出现断电等情况可能导致数据丢失，但这样也提高了性能。
};
```

> LevelDB 的 Put 函数参数是 Slice 类型而不是 std::string
>
> 好处在于 std::string 对于数据进行深拷贝，将字符串的内容复制到一片新的内存，开销较大。而 Slice 在复制时进行浅拷贝，性能高
>

>  Slice 直接使用指针和长度来表示数据，而不持有数据本身。这样，`Slice` 的拷贝只涉及指针和长度的拷贝，而不是复制数据内容。因此，它的拷贝是浅拷贝，而不是深拷贝。  
>

```cpp
class LEVELDB_EXPORT Slice {
private:
    const char* data_;
    size_t size_;
};
 public:
    Slice() : data_(""), size_(0) {}
    Slice(const char* d, size_t n) : data_(d), size_(n) {}
    Slice(const std::string& s) : data_(s.data()), size_(s.size()) {}
    Slice(const char* s) : data_(s), size_(strlen(s)) {}
    Slice(const Slice&) = default;
    Slice& operator=(const Slice&) = default;
    const char* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const char* begin() const { return data(); }
    const char* end() const { return data() + size(); }
    char operator[](size_t n) const {}
    void clear() {}
    void remove_prefix(size_t n) {}
    std::string ToString() const {}
    int compare(const Slice& b) const;
    bool starts_with(const Slice& x) const {}
    inline bool operator==(const Slice& x, const Slice& y) {}
    inline bool operator!=(const Slice& x, const Slice& y) {  }
    inline int Slice::compare(const Slice& b) const {}

} 

```

### 批量写（原子操作）
> sync 结合 WriteBatch 只执行一次 sync 使得开销平摊，并且保证了操作的原子性
>

```cpp
#include "leveldb/writebatch.h"

leveldb::WriteOptions write_opts;
write_opts.sync = true;

leveldb::WriteBatch batch;
batch.Put(keyword, value);
batch.Delete(keyword);

db->Write(write_opts, &batch); # 这里要传入leveldb::WriteBatch对象的指针
```

其实点写操作的背后也调用了 `std::BatchWrite`，如下图：

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736913243785-04e6e298-8edf-4bfd-8743-37ba4684c33f.png)

## 读操作
```cpp
struct LEVELDB_EXPORT ReadOptions {
  bool verify_checksums = false; // 读数据时候是否校验校验和
  bool fill_cache = true; // 将查询结果存到leveldb内部缓存中
};
```

> Get 接口接受三个参数，配置结构体，Slice 类型的 keyword，和一个字符串指针存储返回结果
>

```cpp
virtual Status Get(const ReadOptions& options, const Slice& key,
                   std::string* value) = 0;
```

## 删操作
> Delete 接口 
>

```cpp
virtual Status Delete(const WriteOptions& options, const Slice& key) = 0;
```

## 迭代器
> 迭代器也需要手动销毁，否则编译器会报错，所以也使用智能指针
>
> 迭代器 iter 初始化之后是不可用的，需要改变扫描位置
>
> 迭代器返回的数据是有序的，按照字母序排列
>

```cpp
std::unique_ptr<leveldb::Iterator> iter(db->NewIterator(leveldb::ReadOptions()));
iter->SeekToFirst();
for (; iter->Valid(); iter->Next()) {
    // key 是 leveldb::Slice类型
    auto key = iter->key().ToString();
    auto value = iter->value().ToString();
    std::cout << "key: " << key << " value: " << value << std::endl;
}

for (iter->Seek("b"); iter->Valid(); iter->Next()){
    // iter会定位在大于等于字母序b的下一个元素上
}
```

## 比较器
> 上文提到 LevelDB 默认采用字典序，我们可以继承 Comparator 父类 override 重写虚函数
>
> 每个 LevelDB 文件只能固定使用一个比较器，不能改变
>

```cpp
class LEVELDB_EXPORT Comparator {
 public:
  virtual ~Comparator();
  virtual int Compare(const Slice& a, const Slice& b) const = 0;
  virtual const char* Name() const = 0;
  virtual void FindShortestSeparator(std::string* start, const Slice& limit) const = 0;
  virtual void FindShortSuccessor(std::string* key) const = 0;
};
```

```cpp
class ReverseBytewiseComparator:public leveldb::Comparator{
    int Compare(const leveldb::Slice& a, const leveldb::Slice& b) const override{
        return b.compare(a);
    }
    const char* Name() const override {
        return "customized comparator";
    }
    void FindShortestSeparator(std::string* start, const leveldb::Slice& limit) const override {}
    void FindShortSuccessor(std::string* key) const override {}

};
```



# LevelDB 分析
## 整体架构
![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736675956150-e8228014-6d4e-4f5d-ae85-f83ea49f061e.png)

1. log

LevelDB 使用 WAL 也就是 Write-Ahead-Logging 来保证数据的一致性。当 LevelDB 进行写入操作的时候，会将操作先写入 LOG 日志文件（二进制存储）中（注意是将**操作**写入日志，而不是数据写入日志)。如果直接将数据写入内存，出现断电或者其他异常的时会导致数据丢失。WAL 将操作写入磁盘中的 LOG 日志进行持久化，当系统崩溃时可以重放 LOG 中的操作恢复数据。

2. memTable 和 immutable

当操作被写入磁盘的 LOG 后，数据会被写入内存的 memTable，当 memTable 中积累一定数据就会把这部分数据转为 immutable 并且 new 一个 memTable 以供写入。后台线程会把 immutable 的数据持久化到磁盘，目的是为了限制内存占用。

memTable 这个数据结构使得我们在内存中可以进行高效的读写操作，他是基于 skiplist（跳表）实现的，同时他是有序的，在上文 LevelDB 接口我们定义了 LevelDB 的比较器就是为了自定义排列顺序。

3. sstable

immutable 的数据持久化到磁盘之后称为 level-0 sstable，这个操作称为 mirror compact。sstable 一共有 7 层，随着上层的 sstable 数据增加，会被多路归并进入下一层的 sstable，这个操作称为 major compact。层数越大，数据越多越旧。

多层 sstable 的好处是什么呢？当 immutable 写入磁盘后会形成多个 level-0 sstable，这些持久化文件可能是有交集的。我们可以通过多路合并上层 sstable 文件来减少占用的空间，compact 操作也由后台线程完成。

+ 对 level > 0 的 sstables, 选择其中一个 sstable 与 下一层 sstables 做合并.
+ 对 level = 0 的 sstables, 在选择一个 sstable 后, 还需要找出所有与这个 sstable 有 key 范围重叠的 sstables, 最后统统与level 1 的 sstables 做合并.

> 所以数据的流向是：write -> log(磁盘) -> memTable(内存) -> immutable(内存) -> sstable(磁盘)
>

## 可以做的优化
1. memtable/sstable的阈值size、level--0中的数量阈值、每个高层 level上的总数据量阈值等参数，  
均会影响到compact的运行，从而影响到最终的读写效率，根据不同场景需要做不同的配置，以达  
到最优效果。
2. 内部策略是基于sas盘的I0性能设计，使用其他硬件存储(ssd)时，需要做相应调整。
3. 查找不存在的key一个最坏情况，考虑不同的场景采用写入mock value或者加入bloom filter进  
行优化。
4. db启动后，会将当前的状态写入manifest文件，后面每次compact后，会将产生的操作  
(VersionEdit)作为追加到manifest。如果db实例运行时间很长，mainifest中会有大量的更新  
记录，当db重启时，replay manifest时可能会耗费较长的时间。考虑限制单个manifest中的  
record数量，达到阈值，则做一次rotate。重启时，仅replay最新的manifest即可，做异常情况  
的repair时，则可以连同历史manifest一起replay。
5. LevelDB中除了memtable使用内存有Arena管理外，没有其他的内存管理，内部处理中有很多小对  
象的申请释放，考虑是否可以优化内存使用，当前可以使用tcmalloc。
6. compact时，选取非level-0中符合对应compact_pointer的sstable时，可以使用二分查找定位。



## 数据更新
LevelDB 采用非就地更新而非就地更新，就是在更新数据时不直接在原始数据上进行修改，而是创建一个新的数据副本，并在副本上进行修改，原始数据保持不变，这种机制有利于保证一致性。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736782687704-7f462d60-0ef9-44b4-ad3c-5990b61ea455.png)

LevelDB 采用对每条操作进行编号（SequenceNumber）来实现非就地更新，编号大的操作即为最新操作，如下图：

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736783264680-49792136-3016-408b-9e46-c22f230be8f2.png)

![](https://cdn.nlark.com/yuque/0/2025/svg/49907638/1736911986861-e2be8b1e-354a-4b37-b512-99f372dee747.svg)

LevelDB 进行查询时只要对 SequenceNumber 从高到低排序，第一个操作即为最新操作。

| Key | ValueType | SequenceNumber | Value |
| --- | --- | --- | --- |
| apple | kTypeValue | 1 | red |
| banana | kTypeValue | 2 | yellow |
| orange | kTypeValue | 3 | orange |
| banana | kTypeDeletion | 4 |  |
| apple | kTypeValue | 5 | lightred |


在上述操作下，插入(apple,red)的操作会被插入(apple,lightred)的操作覆盖，插入(banana,yellow)的操作会被删除 banana 的操作覆盖。这种非就地更新存在也存在缺点，leveldb 需要额外的内存空间来存储新副本，此外查询效率也会被影响。leveldb 通过 Compaction 来维护 LsmTree 整体的形状（后文会提及），使得整体的读写放大在合适范围，并且清理掉重复更新删除旧数据。如下图，leveldb 在后台进行 compaction，对数据进行多路合并：

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736784256507-000fac72-b959-4312-9330-c2ea5509ca6d.png)

但是这种 compaction 操作也存在弊端：

1. compation 需要读取文件解压缩，消耗 cpu 和 io 资源。
2. compation 做 rewrite 产生写放大，一定程度上降低磁盘设备的寿命。

## WAL
LevelDB 的日志文件有多个 LogBlock 块组成，每个 `LogBlock` 由多个 `LogRecord` 组成。`LogBlock` 定长而 `LogRecord` 变长，若 `LogRecord` 无法填满则用零字节填充`LogBlock`。

`LogBlock`与`LogRecord`的结构如下：

```cpp
struct LogBlock{
    LogRecord[] records;
    byte[] padding;
};

struct LogRecord{
    uint32 checksum;    // crc校验码
    unint16 length;     // 数据的长度
    byte type;          // Log Record的类型
    byte data[length];  // 实际的数据
};
```

由于单个 `LogBlock` 大小存在限制（默认 32KB），所以一个 `LogBlock` 不一定存的下一条`LogRecord`，这就需要将一个`LogRecord`存储在多个`LogBlock`中，并且使用不同的 `RecordType` 来标识。

> kFullType: 当前 block 保存了所有数据
>
> kFirstType:  当前 block 是 log 中第一个 block，以此类推
>

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736745221882-95decfb4-5cc7-4d47-a5f4-c12184eeb928.png)

假设当前要写入以下三个记录：

> A: length 1000
>
> B: length 97270
>
> C: length 8000
>

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1737475703317-b271799c-39cb-4e1a-b6a0-e6141d9c6904.png)

可以注意到，第三个`LogBLock`插入记录 B 后只剩下 6B 了，所以用零比特填充。

了解了 LevelDB 日志文件的格式，我们从创建`LOG`写入`LOG`到恢复`LOG`全过程来分析 WAL 是如何运作的。

### 创建日志文件
LevelDB 启动时`leveldb::Open`函数调用`DBImpl::Recover`，它会检查目录中是否存在旧的 WAL 文件，通常以 `LOG` 命名存储。假如不存在 WAL 文件，LevelDB 会调用`DBImpl::NewDB` 

`NewDB` 和 `OpenDB` 都涉及数据库的初始化，但两者的主要区别在于它们的用途：

+ `**NewDB**` 用于创建一个全新的数据库。它会创建必要的文件、目录、WAL 文件等。
+ `**OpenDB**` 用于打开一个已经存在的数据库。它会加载已有的数据文件，检查日志文件，恢复未持久化的数据，等等。

再通过`env->NewWritableFile()` 来创建一个新的 WAL 文件。具体的实现可以参考 `db_impl.cc` 文件中的 `Env::NewWritableFile` 函数。`Env::NewWritableFile` 是 LevelDB `Env` 接口中的一个虚拟函数，用于在底层环境中创建一个新的`WritableFile`文件。

在 windows 平台，`NewWritableFile`的实现位于`util/env_windows.cc`：

```cpp
  Status NewWritableFile(const std::string& filename,
                         WritableFile** result) override {
    DWORD desired_access = GENERIC_WRITE;
    DWORD share_mode = 0;  // Exclusive access.
    ScopedHandle handle = ::CreateFileA(
        filename.c_str(), desired_access, share_mode,
        /*lpSecurityAttributes=*/nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
        /*hTemplateFile=*/nullptr);
    if (!handle.is_valid()) {
      *result = nullptr;
      return WindowsError(filename, ::GetLastError());
    }

    *result = new WindowsWritableFile(filename, std::move(handle));
    return Status::OK();
  }
```

它调用底层的 `Windows Api` 创建文件并得到文件句柄，用`ScopedHandle`包装。最后用`WindowsWritableFile`来封装这个句柄，通过指针返回。`ScopedHandle`作用类似智能指针，当程序意外退出时， `ScopedHandle` 对象离开作用域，析构函数会自动关闭文件句柄。

`WritableFile` 是 LevelDB 提供的一个抽象类，通常由不同的操作系统平台实现，它的具体实现有多个，在 `util/env_posix.cc` 和 `util/env_windows.cc` 中。这一抽象类的主要职责是提供文件写入操作（如 `Append()`, `Sync()`, `Close()` 等）接口。

在 LevelDB 的源码中，`WritableFile` 的声明位于 `include/leveldb/env.h` 文件中：

```cpp
class LEVELDB_EXPORT WritableFile {
    public:
        WritableFile() = default;
        WritableFile(const WritableFile&) = delete;
        WritableFile& operator=(const WritableFile&) = delete;
        virtual ~WritableFile();
        virtual Status Append(const Slice& data) = 0;
        virtual Status Close() = 0;
        virtual Status Flush() = 0;
        virtual Status Sync() = 0;
};
```

`WindowsWritableFile` 是 `WritableFile` 的具体实现类，专门用于 Windows 平台。它实现了 `WritableFile` 中定义的纯虚函数，使用 Windows API 进行文件操作。`WindowsWritableFile`的实现位于`util/env_windows.cc`中：

```cpp
class WindowsWritableFile : public WritableFile {
 public:
  WindowsWritableFile(std::string filename, ScopedHandle handle)
      : pos_(0), handle_(std::move(handle)), filename_(std::move(filename)) {}

  ~WindowsWritableFile() override = default;

  Status Append(const Slice& data) override {
    size_t write_size = data.size();
    const char* write_data = data.data();

    // Fit as much as possible into buffer.
    size_t copy_size = std::min(write_size, kWritableFileBufferSize - pos_);
    std::memcpy(buf_ + pos_, write_data, copy_size);
    write_data += copy_size;
    write_size -= copy_size;
    pos_ += copy_size;
    if (write_size == 0) {
      return Status::OK();
    }

    // Can't fit in buffer, so need to do at least one write.
    Status status = FlushBuffer();
    if (!status.ok()) {
      return status;
    }

    // Small writes go to buffer, large writes are written directly.
    if (write_size < kWritableFileBufferSize) {
      std::memcpy(buf_, write_data, write_size);
      pos_ = write_size;
      return Status::OK();
    }
    return WriteUnbuffered(write_data, write_size);
  }

  Status Close() override {
    Status status = FlushBuffer();
    if (!handle_.Close() && status.ok()) {
      status = WindowsError(filename_, ::GetLastError());
    }
    return status;
  }

  Status Flush() override { return FlushBuffer(); }

  Status Sync() override {
    // On Windows no need to sync parent directory. Its metadata will be updated
    // via the creation of the new file, without an explicit sync.

    Status status = FlushBuffer();
    if (!status.ok()) {
      return status;
    }

    if (!::FlushFileBuffers(handle_.get())) {
      return Status::IOError(filename_,
                             GetWindowsErrorMessage(::GetLastError()));
    }
    return Status::OK();
  }

 private:
  Status FlushBuffer() {
    Status status = WriteUnbuffered(buf_, pos_);
    pos_ = 0;
    return status;
  }

  Status WriteUnbuffered(const char* data, size_t size) {
    DWORD bytes_written;
    if (!::WriteFile(handle_.get(), data, static_cast<DWORD>(size),
                     &bytes_written, nullptr)) {
      return Status::IOError(filename_,
                             GetWindowsErrorMessage(::GetLastError()));
    }
    return Status::OK();
  }

  // buf_[0, pos_-1] contains data to be written to handle_.
  char buf_[kWritableFileBufferSize];
  size_t pos_;

  ScopedHandle handle_;
  const std::string filename_;
};
```

现在分析一下`WindowsWritableFile`是如何实现的。

LevelDB 并不是每次写入都直接写一个数据块到文件，而是通过缓冲的机制来提高写入效率。具体来说，LevelDB 会使用一个**缓冲区**（如 `WindowsWritableFile` 类中的 `buf_`）来缓存数据，然后再一次性写入文件。这种做法能够显著提高写入性能，因为频繁的磁盘操作是非常低效的。  

1. `Append`方法会计算当前缓冲区剩余的空间，并且将小等于缓冲区剩余空间大小的数据写入。如果此时数据全部写入则返回`Status::OK`，如果数据无法完全写入缓冲区，那么它会调用`FlushBuffer`方法，将缓冲区的数据写入文件，之后再把剩余数据写入缓冲区。`FlushBuffer`函数底层也是通过`WriteUnbuffered`函数调用`Windows Api`写入缓冲区数据到文件。

```cpp
Status WriteUnbuffered(const char* data, size_t size) {
    DWORD bytes_written;
    if (!::WriteFile(handle_.get(), data, static_cast<DWORD>(size),
                     &bytes_written, nullptr)) {
      return Status::IOError(filename_,
                             GetWindowsErrorMessage(::GetLastError()));
    }
    return Status::OK();
}
```

2. `Close`方法会调用`FlushBuffer`方法将缓冲区数据写入磁盘，并且关闭文件句柄。
3. `Sync`方法与`Flush`方法都是将内存缓冲区中的数据写入文件，两者有什么区别呢？

这里我们要了解 windows 系统写入数据的机制。在 Windows 系统中，当你向文件写入数据时，数据可能先存储在内核的缓冲区中（操作系统的文件缓存）。操作系统会定期将这些缓冲区中的数据写入磁盘，而不是每次都立即写入。  而`Sync`方法除了调用`FlushBuffer`方法，还调用了`Windows Api`` ::FlushFileBuffers()`。它会将文件句柄（`handle_`）指向的文件缓冲区中的所有数据强制刷新到磁盘，从而确保文件的内容一定被持久化。  

### 追加日志文件
调用`db->Put()`或`db->Delete()`等接口时， LevelDB 会先将操作记录到 `LOG`，然后将相同的数据写入到内存中的`MemTable`中。接下来我以 `db->Put()`操作，分析 LevelDB 是如何写入日志的。

```cpp
Status DB::Put(const WriteOptions& opt, const Slice& key, const Slice& value) {
    WriteBatch batch;
    batch.Put(key, value);
    return Write(opt, &batch);
}
```

`db->Put()`将待写入的键值对封装到`WriteBatch`中，通过再调用`Write`方法写入（前文提到过，`WriteBatch`能够实现批量写入的原子操作，保证数据一致性）。

`Write`方法实现较为复杂，设计写入队列、WAL、更新`MemTable`等操作，这里我们主要研究如何写入日志的。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1737702329809-29cb30b3-fa8d-4208-bc40-645691f3fd2e.png)

我们暂时不关心`write_batch`和`WriteBatchInternal::Contents`是如何实现的，这里 LevelDB 将待写入的数据转换为`Slice`传给`log_->AddRecord()`这个方法。`log_`是	`log::Writer`的实例，`Log` 类是负责管理日志的核心类，通过 `Log` 类和 `WritableFile` 接口实现文件的具体写入操作。

<font style="color:rgb(37, 41, 51);">LevelDB会为每次写入封装一个Writer 对象，这个对象定义在`db/log_writer.h`。

```cpp
class Writer {
 public:
  explicit Writer(WritableFile* dest);

  // Create a writer that will append data to "*dest".
  // "*dest" must have initial length "dest_length".
  // "*dest" must remain live while this Writer is in use.
  Writer(WritableFile* dest, uint64_t dest_length);

  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;

  ~Writer();

  Status AddRecord(const Slice& slice);

 private:
  Status EmitPhysicalRecord(RecordType type, const char* ptr, size_t length);

  WritableFile* dest_;
  int block_offset_;  // Current offset in block

  // crc32c values for all supported record types.  These are
  // pre-computed to reduce the overhead of computing the crc of the
  // record type stored in the header.
  uint32_t type_crc_[kMaxRecordType + 1];
};
```

`<font style="color:rgb(37, 41, 51);">AddRecord`<font style="color:rgb(37, 41, 51);">方法主要就是对当前写入的 `<font style="color:rgb(37, 41, 51);">LogRecord`<font style="color:rgb(37, 41, 51);">做切割成多个 `<font style="color:rgb(37, 41, 51);">Fragement`<font style="color:rgb(37, 41, 51);">，之后调用`<font style="color:rgb(37, 41, 51);">EmitPhysicalRecord`<font style="color:rgb(37, 41, 51);">将数据写入磁盘。

```cpp
Status Writer::AddRecord(const Slice& slice) {
  const char* ptr = slice.data();
  size_t left = slice.size();

  Status s;
  bool begin = true;
  do {
    const int leftover = kBlockSize - block_offset_;
    assert(leftover >= 0);
    if (leftover < kHeaderSize) {
      if (leftover > 0) {
        static_assert(kHeaderSize == 7, "");
        dest_->Append(Slice("\x00\x00\x00\x00\x00\x00", leftover));
      }
      block_offset_ = 0;
    }
    assert(kBlockSize - block_offset_ - kHeaderSize >= 0);

    const size_t avail = kBlockSize - block_offset_ - kHeaderSize;
    const size_t fragment_length = (left < avail) ? left : avail;

    RecordType type;
    const bool end = (left == fragment_length);
    if (begin && end) {
      type = kFullType;
    } else if (begin) {
      type = kFirstType;
    } else if (end) {
      type = kLastType;
    } else {
      type = kMiddleType;
    }

    s = EmitPhysicalRecord(type, ptr, fragment_length);
    ptr += fragment_length;
    left -= fragment_length;
    begin = false;
  } while (s.ok() && left > 0);
  return s;
}
```

前文说过，若 `LogRecord` 无法填满 `LogBlock`，则用零字节填充。 `AddRecord`方法的思路就是先判断当前`LogBlock`剩余空间是否大于`kHeaderSize`（7B 日志头部），如果空间不足则用零字节填充。之后再判断`LogBlock`剩余空间是否能够存储数据从而决定`ValueType`，调用`EmitPhysicalRecord`方法将数据写入磁盘。

```cpp
Status Writer::EmitPhysicalRecord(RecordType t, const char* ptr,
                                  size_t length) {
  assert(length <= 0xffff);  // Must fit in two bytes
  assert(block_offset_ + kHeaderSize + length <= kBlockSize);

  // Format the header
  char buf[kHeaderSize];
  buf[4] = static_cast<char>(length & 0xff);
  buf[5] = static_cast<char>(length >> 8);
  buf[6] = static_cast<char>(t);

  // Compute the crc of the record type and the payload.
  uint32_t crc = crc32c::Extend(type_crc_[t], ptr, length);
  crc = crc32c::Mask(crc);  // Adjust for storage
  EncodeFixed32(buf, crc);

  // Write the header and the payload
  Status s = dest_->Append(Slice(buf, kHeaderSize));
  if (s.ok()) {
    s = dest_->Append(Slice(ptr, length));
    if (s.ok()) {
      s = dest_->Flush();
    }
  }
  block_offset_ += kHeaderSize + length;
  return s;
}
```

`EmitPhysicalRecord`方法会对传入的数据计算 CRC 校验码，并且调用`WritableFile`实例`dest_`的写接口将数据写入磁盘。

### 回放日志文件
之前说过，LevelDB 启动时 `DBImpl::Recover`检查目录中是否存在旧的 WAL 文件。假如不存在 WAL 文件，LevelDB 会调用`DBImpl::NewDB`。假如存在 WAL 文件，`versions_->Recover(save_manifest)`会回放日志文件，这个部分会在 `崩溃控制`** **细说。

## 快照查询
levelDB 支持批量写入的原子操作但是不支持批量查询的原子操作，这就导致进行读操作的时候可能出现不一致的情况，如下图：

> 这种情况下可能出现 student1 读取的是 xiaoming，student2 读取的是 xiaoB
>
> 原因在于两次查询分别调用 db->Get()是独立的
>

```cpp
db->Put(leveldb::WriteOptions, "student1", "xiaoming");
db->Put(leveldb::WriteOptions, "student2", "xiaowang");

std::thread t1([db](){
    db->Get(leveldb::ReadOptions, "student1", &student1_info);
    db->Get(leveldb::ReadOptions, "student2", &student2_info);
})

std::thread t2([db](){
    db->Put(leveldb::WriteOptions, "student1", "xiaoA");
    db->Put(leveldb::WriteOptions, "student2", "xiaoB");
})

t1.join();
t2.join();
```

解决方案就是基于**快照查询**，下图为 `leveldb::Options` 中的参数 snapshot：

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1736752964686-f0aadc7f-0684-42b9-a161-7f9084192466.png)

```cpp
db->Put(leveldb::WriteOptions, "student1", "xiaoming");
db->Put(leveldb::WriteOptions, "student2", "xiaowang");

std::thread t1([db](){
    auto read_opts = leveldb::ReadOptions();
    const leveldb::SnapShot* snapshot = db->GetSnapshot();  // 采用此时的快照查询
    read_opts.snapshot = snapshot; 
    db->Get(read_opts, "student1", &student1_info);
    db->Get(read_opts, "student2", &student2_info);
})

std::thread t2([db](){
    db->Put(leveldb::WriteOptions, "student1", "xiaoA");
    db->Put(leveldb::WriteOptions, "student2", "xiaoB");
})

t1.join();
t2.join();
```

`db->GetSnapshot()` 分配的快照位于堆上，所以我们还需要手动释放。`leveldb::Snapshot` 的析构函数被 delete 了，提供了专门的释放函数 `db->ReleaseSnapshot()`。为了实现类似 smartpointer 的智能管理效果，我们可以封装一个类似的 guard 来管理快照。

```cpp
class SnapshotGuard{
private:
    leveldb::DB *db_{nullptr};
    const leveldb::Snapshot *snapshot_{nullptr};
public:
    SnapshotGuard(leveldb::DB db): db_(db), snapshot(db->GetSnapshot()){}
    SnapshotGuard(const SnapshotGuard&) = delete;
    SnapshotGuard operate=(const SnapshotGuard&) = delete;
    ~SnapshotGuard(){
        db_->ReleaseSnapshot(snapshot_);
    }
    const leveldb::Snapshot* GetSnapshot() const{
        return snapshot_;
    }
};


int main(){
    SnapshotGuard guard(db.get());
    auto read_opts = leveldb::ReadOptions();
    read_opts.snapshot = guard.GetSnapshot();
}
```

那 leveldb 是如何实现快照这种功能的呢？是不是保存了快照时全部数据的副本呢？其实 leveldb 实现快照的方式非常简单，只需要保存快照时最大的 SequenceNumber 即可，进行查询操作时从 Max_SequenceNumber 向前遍历就是快照时的数据。

## SkipList
上文说到 leveldb 将 memTable 和 unmutable memTable 存储在内存中，将 sstable 存储在磁盘中，那 leveldb 是如何操作内存中的数据（memtable）呢？它是借助了跳表 SkipList 这一数据结构，我们借助一个例子來了解这个数据结构的作用。

假如我们拥有一个 1~10 的有序链表如下：

```plain
1->2->3->4->5->6->7->8->9->10
```

在这个有序链表中插入和查找操作时间复杂度都是 O(n)，由于它是链表所以不能二分查找，那有没有办法借助二分的思想更快的查找呢？我们可以提取节点作索引如下：

```plain
1 ->  3 ->  5 ->  7 ->  9
1->2->3->4->5->6->7->8->9->10
```

此时如果我们要访问 8，可以 1->3->5->7->8，减少了查询所需的次数，尤其在数据量大的情况下效果更加明显，并且我们可以建立多层索引來优化，时间复杂度是 O(logn)。

多层索引跳表如下(一般采用每两个节点提取一个当索引的原则)：

```plain
1->   ->    5    ->     9      ->      13      ->      17 
1 ->  3 ->  5 ->  7 ->  9  ->  11  ->  13  ->  15  ->  17
1->2->3->4->5->6->7->8->9->10->11->12->13->14->15->16->17
```

现在还存在一个问题，skiplist 该怎么插入数据呢？我们可以类似通过查找操作得到插入位置，那节点更新后上层索引需要改变吗？这里 skiplist 采用动态更新（概率），使每个节点被提取到上层的概率是 1/2（原始链表提取到一级索引概率是 1/2，提取到二级索引是 1/4 以此类推），这也等价于每两个节点提取一个当索引。

现在我们就可以手写一个 skiplist 了，首先从定义数据结构开始：

```cpp
typedef struct SkiplistNode{
    std::string key, data;
    struct SkiplistNode levels[SKIPLIST_LEVEL];
};

typedef struct Skiplist{
    SkiplistNode *head;
    unsigned long length;
    int level;
}list;
```

将传统链表节点的 next 指针改为 next 指针数组，索引 i 的 next 指针对于第 i 层的 下一个节点指针。

```cpp
// 在每一层进行有序单链表的查找
// 当查找进行不下去时level--进入下一层
bool find(const std::string& key, std::string& value){
    SkipListNode* cur = this->_head;
    for (int i=this->_level-1; i>=0; i--){
        while (cur->next[i] && cur->next[i]->key < key){
            cur = cur->next[i];
        }
    }
    // 此时得到的cur应该是level[0]目标节点的prev
    if (cur->next[0] && cur->next[0]->key == key){
        value = cur->next[0]->value;
        return true;
    }
    return false;
}
```

插入操作相对复杂一些，需要找到每一层的前驱，在每一层进行单链表插入。

```cpp
#include <random>

std::random_device rd;
std::mt19937 gen(rd()); 
std::uniform_int_distribution<> dis(0, 1);


int RandLevel(){
    int level = 0;
    for (;;level++){
        if (dis(gen) == 0){
            return level;
        }
    }
}


void add(const std::string& key, const std::string& value){
        SkipListNode* prev[MAX_LEVELS];
        SkipListNode* cur = this->_head;
        for (int i=this->_level-1; i>=0; i--){
            while (cur->next[i] && cur->next[i]->key < key){
                cur = cur->next[i];
            }
            prev[i] = cur;
        }
        int newLevel = this->_RandLevel();
        SkipListNode* newNode = new SkipListNode;
        newNode->key = key;
        newNode->value = value;

        for (int i=newLevel-1; i>=0 ;i++){
            // 将新节点的next指向prev的next
            newNode->next[i] = prev[i]->next[i];
            // 将prev的next指向新节点
            prev[i]->next[i] = newNode;
        }
    }
```

了解了跳表的思想和每个功能的函数基本实现，还存在着一些细节问题：

1. 每次插入时节点有 (1/2)^k 的概率被插入第 k 层，如果新节点被插入第 n 层，而当前 skiplist 的 level 为 n-2 该怎么办呢？
2. 在 skiplist 中都是存储着节点的指针，也就是将节点存放在堆上，该如何管理这块内存？ 

```cpp
#include <iostream>
#include <random>
#define MAX_LEVELS 50

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, 1);

class SkipList{
private:
    struct SkipListNode{
        std::string key, value;
        SkipListNode* next[MAX_LEVELS] = {nullptr};
        SkipListNode() = default;
        SkipListNode(const std::string& _key): key(_key){}
    };
    int _level;
    unsigned long _length;
    SkipListNode *_head;
    int _RandLevel(){
        int level = 1;
        for (;;level++){
            if (dis(gen) == 0){
                return level;
            }
        }
    }


public:
    SkipList(){
        this->_level = 1;
        this->_length = 0;
        this->_head = new SkipListNode("head");
    }
    ~SkipList(){
        SkipListNode* cur = this->_head;
        while(cur){
            SkipListNode* next = cur->next[0];
            delete cur;
            cur = next;
        }
    }
    bool find(const std::string& key, std::string& value){
        SkipListNode* cur = this->_head;
        for (int i=this->_level-1; i>=0; i--){
            while (cur->next[i] && cur->next[i]->key < key){
                cur = cur->next[i];
            }
        }
        // 此时得到的cur应该是level[0]目标节点的prev
        if (cur->next[0] && cur->next[0]->key == key){
            value = cur->next[0]->value;
            return true;
        }
        return false;
    }

    void add(const std::string& key, const std::string& value){
        SkipListNode* prev[MAX_LEVELS] = {nullptr};
        SkipListNode* cur = this->_head;
        for (int i=this->_level-1; i>=0; i--){
            while (cur->next[i] && cur->next[i]->key < key){
                cur = cur->next[i];
            }
            prev[i] = cur;
        }
        int newLevel = this->_RandLevel();
        for (int i=this->_level;i<newLevel;i++){
            prev[i]=this->_head;
        }
        this->_level = std::max(this->_level, newLevel);

        SkipListNode* newNode = new SkipListNode;
        newNode->key = key;
        newNode->value = value;

        for (int i=newLevel-1; i>=0 ;i--){
            // 将新节点的next指向prev的next
            newNode->next[i] = prev[i]->next[i];
            // 将prev的next指向新节点
            prev[i]->next[i] = newNode;
        }
    }

    void display(){
        for (int i=this->_level;i>=0;i--){
            for (SkipListNode* node=this->_head;node != nullptr;node=node->next[i]){
                std::cout << " -> " << node->key;
            }
            std::cout << std::endl;
        }
    }
};


int main(){
    // test
    SkipList skiplist;
    skiplist.add("stu1", "a");
    skiplist.add("stu2", "b");
    skiplist.add("stu3", "c");
    skiplist.add("stu4", "d");
    skiplist.add("stu5", "e");
    skiplist.add("stu6", "f");
    std::string value;
    std::cout << "Search for key stu1" << (skiplist.find("stu1", value) ? "Found, value=" + value : "Not found");
    std::cout << "Search for key stu2" << (skiplist.find("stu2", value) ? "Found, value=" + value : "Not found");
    skiplist.display();
}
```

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1737031880910-3f56207a-7c4d-4a7e-9d51-a6f585c14168.png)

> 改进空间
>

1. 有没有办法优化 skiplist 的查询方式？在特殊情况下可能出现 多个 level 查找结果相同能不能优化。例如在上述案例中 level2~level7 都出现了 head->stu2 。
2. RandLevel 函数能否优化提高效率。

> LevelDB 中的 skiplist 与普通 skiplist 有什么区别？
>

1. leveldb 的 skiplist 是线程安全的，支持多线程操作。
2. leveldb 插入新节点选择的 level 不是完全随机，而是基于 key 的哈希值，有助于保证跳表的平衡性。
3. leveldb 的 skiplist 提供迭代器，支持范围查询和顺序遍历。
4. leveldb 的 skiplist 不用 new 和 delete 操作节点，而是通过内存池 arena 分配节点，删除时通过延迟回收策略，避免在高并发情况下出现内存问题。

现在来看一下 leveldb 中是如何实现 skiplist 的，leveldb::Skiplist 和 leveldb::Iterator 声明如下:

```cpp
template <typename Key, class Comparator>
class SkipList {
private:
    struct Node;

public:
  explicit SkipList(Comparator cmp, Arena* arena);
  SkipList(const SkipList&) = delete;
  SkipList& operator=(const SkipList&) = delete;
  void Insert(const Key& key);
  bool Contains(const Key& key) const;

  class Iterator {
   public:
    explicit Iterator(const SkipList* list);
    bool Valid() const;
    const Key& key() const;
    void Next();
    void Prev();
    void Seek(const Key& target);
    void SeekToFirst();
    void SeekToLast();
   private:
    const SkipList* list_;
    Node* node_;
  };

 private:
  enum { kMaxHeight = 12 };

  inline int GetMaxHeight() const {
    return max_height_.load(std::memory_order_relaxed);
  }

  Node* NewNode(const Key& key, int height);
  int RandomHeight();
  bool Equal(const Key& a, const Key& b) const { return (compare_(a, b) == 0); }
  bool KeyIsAfterNode(const Key& key, Node* n) const;
  Node* FindGreaterOrEqual(const Key& key, Node** prev) const;
  Node* FindLessThan(const Key& key) const;
  Node* FindLast() const;
  Comparator const compare_;
  Arena* const arena_; 
  Node* const head_
  std::atomic<int> max_height_; 
  Random rnd_;
};

```

leveldb 的 skiplist 类只插入不修改或删除，MemTable的修改或删除是通过插入有响应标识或序号的key实现的。进一步看看 `contains` 和 `insert` 是如何实现的：

```cpp
template <typename Key, class Comparator>
bool SkipList<Key, Comparator>::Contains(const Key& key) const {
    Node* x = FindGreaterOrEqual(key, nullptr);
    if (x != nullptr && Equal(key, x->key)) {
        return true;
    } else {
        return false;
    }
}

template <typename Key, class Comparator>
void SkipList<Key, Comparator>::Insert(const Key& key) {
  Node* prev[kMaxHeight];
  Node* x = FindGreaterOrEqual(key, prev);

  assert(x == nullptr || !Equal(key, x->key));

  int height = RandomHeight();
  if (height > GetMaxHeight()) {
    for (int i = GetMaxHeight(); i < height; i++) {
      prev[i] = head_;
    }
    max_height_.store(height, std::memory_order_relaxed);
  }

  x = NewNode(key, height);
  for (int i = 0; i < height; i++) {
    x->NoBarrier_SetNext(i, prev[i]->NoBarrier_Next(i));
    prev[i]->SetNext(i, x);
  }
}


template <typename Key, class Comparator>
typename SkipList<Key, Comparator>::Node*
SkipList<Key, Comparator>::FindGreaterOrEqual(const Key& key,
                                              Node** prev) const {
  Node* x = head_;
  int level = GetMaxHeight() - 1;
  while (true) {
    Node* next = x->Next(level);
    if (KeyIsAfterNode(key, next)) {
      x = next;
    } else {
      if (prev != nullptr) prev[level] = x;
      if (level == 0) {
        return next;
      } else {
        level--;
      }
    }
  }
} 
```

1. `std::atomic` 是 C++11 引入的一个模板类，用于提供原子操作。leveldb 用 atomic 管理 node 和 max_level，从而保证线程安全。通过原子类实现的Node，SkipList能够保证“读读并发”、“读写并发”的线程安全。
2.  创建新节点是通过 `std::arena` 分配内存的，而不是用关键字 new，后面会仔细研究 `leveldb::arena`。

总体来看， leveldb 将查找每一层前驱的操作和 `find` 操作都封装在`FindGreaterOrEqual` 这个函数里，`insert` 和 `contains` 的整体思路和传统 skiplist 相似。

## 内存分配器Arena
在上文我们提到，leveldb 在`leveldb::Skiplist`和 `leveldb::MemTable`分配内存是通过 `leveldb::Arena`，而不是通过 native 的 `new/delete`或者`malloc/free`关键字，这是为什么呢？

+ 在`leveldb::Skiplist`和 `leveldb::MemTable`分配内存操作非常频繁，如果每一次都调用`new`插入一个小小的键值对开销非常大，会产生很多的内存碎片。`leveldb::Arena`的解决方案是先分配出一块大内存`Block`，之后每次从这块大内存中取一部分分配。这种方法只有第一次分配内存时存在开销，后面只需要改变指针即可。
+ native 的关键词只提供了分配删除内存的简单操作，`leveldb::Arena`使得 leveldb 可以对分配的内存生命周期进行管理，当`leveldb::Arena`被销毁时分配的所有内存可以被直接释放。
+ `leveldb::Arena`可以跟踪记录内存的使用情况。

`leveldb::Arena`的声明如下:

```cpp
class Arena {
public:
	Arena(): alloc_ptr_(nullptr), alloc_bytes_remaining_(0), memory_usage_(0) {}
	Arena(const Arena&) = delete;
	Arena& operator=(const Arena&) = delete;
	~Arena() {
	  for (size_t i = 0; i < blocks_.size(); i++) {
		  delete[] blocks_[i];
	  }
	}
	char* Allocate(size_t bytes)
	char* AllocateAligned(size_t bytes);
	size_t MemoryUsage() const {
	    return memory_usage_.load(std::memory_order_relaxed);
	  }
private:
	char* AllocateFallback(size_t bytes);
	char* AllocateNewBlock(size_t block_bytes);
	char* alloc_ptr_;
	size_t alloc_bytes_remaining_;
	std::vector<char*> blocks_;
	std::atomic<size_t> memory_usage_;
};
```

`leveldb::Arena`对外提供了两个函数`Allocate`和`AllocateAligned`来分配内存，分别是不考虑内存对齐和内存对齐的版本。`leveldb::Arena`在每次创建 block 时会将其插入`blocks_`数组中，在析构函数中一并释放。这也是由于 `leveldb::MemTable`没有实际删除操作的原因，它只需要在将`MemTalbe`dump 进`SSTable`时释放整个 `Arena`就好了。

```cpp
inline char* Arena::Allocate(size_t bytes) {
  assert(bytes > 0);
  if (bytes <= alloc_bytes_remaining_) {
    char* result = alloc_ptr_;
    alloc_ptr_ += bytes;
    alloc_bytes_remaining_ -= bytes;
    return result;
  }
  return AllocateFallback(bytes);
}

```

`leveldb::Arena`的逻辑如下：

1. 构造函数将可分配内存的容量初始化为 0。
2. `leveldb::Arena::Allocate`函数判断，若当前 `block`剩余内存大于申请内存大小则修改指针和剩余内存大小；若剩余内存大小不足， 则调用`leveldb::Arena::AllocateFallback`创建新`block`并且分配内存。

```cpp
char* Arena::AllocateFallback(size_t bytes) {
  if (bytes > kBlockSize / 4) {
    char* result = AllocateNewBlock(bytes);
    return result;
  }
  alloc_ptr_ = AllocateNewBlock(kBlockSize);
  alloc_bytes_remaining_ = kBlockSize;

  char* result = alloc_ptr_;
  alloc_ptr_ += bytes;
  alloc_bytes_remaining_ -= bytes;
  return result;
}

char* Arena::AllocateNewBlock(size_t block_bytes) {
  char* result = new char[block_bytes];
  blocks_.push_back(result);
  memory_usage_.fetch_add(block_bytes + sizeof(char*),
                          std::memory_order_relaxed);
  return result;
}
```

`leveldb::Arena::AllocateFallback`的设计思路如下：如果申请的内存较大（大于 1/4 的 block 大小）那么直接向 OS 申请，也就是直接分配一个 bytes 大小的 block；如果申请的内存较小，那么申请一块新的 block ，从 block 中分割 bytes 大小的内存分配。这么做的好处在于：

+ 申请大内存的几率比较小，不会很频繁，找os要虽然慢但是可以避免内存碎片。
+ 申请小内存的几率大，会比较频繁，从block中分配，效率高并且碎片也少。

如果学过计算机组成原理应该明白，计算机每次读取一个字长的数据。假如我将 2B 的数据存储在 07H 和 08H，那么 CPU 就需要取两次内存（一次取一行）。`leveldb::Arena::AllocateAligned`的定义如下。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1737272060879-466ad44b-6aa8-4d7f-bbf4-e362826ec260.png)

```cpp
char* Arena::AllocateAligned(size_t bytes) {
  const int align = (sizeof(void*) > 8) ? sizeof(void*) : 8;
  static_assert((align & (align - 1)) == 0,
                "Pointer size should be a power of 2");
  size_t current_mod = reinterpret_cast<uintptr_t>(alloc_ptr_) & (align - 1);
  size_t slop = (current_mod == 0 ? 0 : align - current_mod);
  size_t needed = bytes + slop;
  char* result;
  if (needed <= alloc_bytes_remaining_) {
    result = alloc_ptr_ + slop;
    alloc_ptr_ += needed;
    alloc_bytes_remaining_ -= needed;
  } else {
    result = AllocateFallback(bytes);
  }
  assert((reinterpret_cast<uintptr_t>(result) & (align - 1)) == 0);
  return result;
}
```

`leveldb::Arena::AllocateAligned`首先通过`alloc_ptr % align`得到余数，这样就可以知道为了使得内存对齐需要占用多少 B 也就是`needed`。如果占用空间小于一个 `block` ， 那么就需要将它向后移动，对齐到 `align`的整数倍；如果占用空间大于一个 `block` 那么就是对齐的，直接分配一个`block`即可。函数的思路比较简单，但是实现相对有点复杂。

1. `const int align = (sizeof(void*) > 8) ? sizeof(void*) : 8;`获取当前机器的字长，如果当前平台的字长大于8，则对齐长度为字长。
2. `(align & (align - 1)) == 0`判断是否为 2 的幂次。当 `align` 是 2 的幂时，它的二进制表示只有一个 `1`，当减去 1 后，二进制中的 `1` 会变成 `0`，并且原先 `1` 后面所有的 `0` 都变成 `1`。所以，`x` 和 `x - 1` 的按位与操作（`&`）会使得所有位都为 0，即结果为 0。  
3. `reinterpret_cast<uintptr_t>(alloc_ptr_) & (align - 1)`等价于`alloc_ptr % align`

## MemTable
Memtable 对 Skiplist 进行封装。leveldb::Skiplist 是一个只有 key 的查找数据结构，仅提供了插入查找功能和简单的迭代器。`leveldb::memtable` 在此基础上 对外提供了key/value的增删改查操作，还提供了正向迭代器与反向迭代器。`leveldb::memtable` 的声明如下：

```cpp
class InternalKeyComparator;
class MemTableIterator;

class MemTable {
public:
    explicit MemTable(const InternalKeyComparator& comparator);
    MemTable(const MemTable&) = delete;
    MemTable& operator=(const MemTable&) = delete;
    void Ref() { ++refs_; }
    void Unref() {
        --refs_;
        assert(refs_ >= 0);
        if (refs_ <= 0) {
            delete this;
        }
    }
    size_t ApproximateMemoryUsage();
    Iterator* NewIterator();
    void Add(SequenceNumber seq, ValueType type, const Slice& key,
    const Slice& value);
    bool Get(const LookupKey& key, std::string* value, Status* s);

    private:
    friend class MemTableIterator;
    friend class MemTableBackwardIterator;

    struct KeyComparator {
    const InternalKeyComparator comparator;
    explicit KeyComparator(const InternalKeyComparator& c) : comparator(c) {}
    int operator()(const char* a, const char* b) const;
};

    typedef SkipList<const char*, KeyComparator> Table;

    ~MemTable(); 

    KeyComparator comparator_;
    int refs_;
    Arena arena_;
    Table table_;
};

```

### 引用计数
 	引用计数是一种内存管理技术，主要用于跟踪对象或资源的引用次数。在 LevelDB 中，MemTable 使用了引用计数来避免在多个组件之间传递数据时进行不必要的拷贝，它能确保在最后一个引用释放时数据才会被销毁，这样可以减少内存的开销和拷贝的开销从而提高性能。

### Key 到 Key/Value
前面我们说到 `leveldb::Skiplist` 只能存储 key，而 `leveldb::MemTable` 能存储键值对，他是怎么实现的呢？`leveldb::MemTable` 将 key 和 value 组合成一个新的 key 存储在 `leveldb::Skiplist` 中。Key/Value 键值对存储的格式如下：

![](https://cdn.nlark.com/yuque/0/2025/svg/49907638/1737097601126-8b7a6f1f-bcc4-4dd9-8c5b-19f9deef03c6.svg)

```cpp
void MemTable::Add(SequenceNumber s, ValueType type, const Slice& key,
                   const Slice& value) {
  // Format of an entry is concatenation of:
  //  key_size     : varint32 of internal_key.size()
  //  key bytes    : char[internal_key.size()]
  //  tag          : uint64((sequence << 8) | type)
  //  value_size   : varint32 of value.size()
  //  value bytes  : char[value.size()]
  size_t key_size = key.size();
  size_t val_size = value.size();
  size_t internal_key_size = key_size + 8;
  const size_t encoded_len = VarintLength(internal_key_size) +
                             internal_key_size + VarintLength(val_size) +
                             val_size;
  char* buf = arena_.Allocate(encoded_len);
  char* p = EncodeVarint32(buf, internal_key_size);
  std::memcpy(p, key.data(), key_size);
  p += key_size\
      9
  EncodeFixed64(p, (s << 8) | type);
  p += 8;
  p = EncodeVarint32(p, val_size);
  std::memcpy(p, value.data(), val_size);
  assert(p + val_size == buf + encoded_len);
  table_.Insert(buf);
}

```

`MemTable::Add`函数向 `std::arena `申请了一块 `SkiplistKey` 大小(encoded_len)的内存，然后通过 `std::memcpy`将每个部分拷贝进内存，最后调用 `std::skiplist` 的 `insert` 方法插入。在函数中 leveldb 用`size_t`类型定义 key 长度相关的变量，它调用了`EncodeVarint32`函数将一个 **固定长度（size_t 是 c++的无符号**32 位整数类型**）** 的变量编码为**可变长度**的 Varint 格式字节序列。例如 `internal_key_size`的值为 300，那么他的二进制表示为100101100，如果使用 `EncodeVarint32` 编码后，会生成 2 个字节的 Varint 格式数据（下个章节会再讨论）：

+ 第一个字节：`10101100`（低 7 位是 `0101100`，最高位为 1，表示还有后续字节）。
+ 第二个字节：`00000010`（低 7 位是 `0000010`，最高位为 0，表示结束）。

既然我们知道了插入操作，删除和修改操作实现也是同理，只要插入新的记录即可，用不同的 `ValueType`区分修改或删除。那查找操作该如何实现呢？我们只有 `UserKey`但是没有插入时的`SequenceNumber`，没有办法构造出相同的`SkiplistKey`。leveldb 的方法是通过`leveldb::LookupKey`类组装一个新的 `SkiplistKey`再进行比较。

> PS:我们再明确一下在 `leveldb::MemTable`中各种 key 的定义
>

    - `UserKey` 是用户插入的键值
    - `InternalKey` 包含了键值长度与`SequenceNumber`和`ValueType`
    - `SkiplistKey` 是 leveldb 将 KeyValue 组合而成得到的新键值
    - `leveldb::LookupKey` 是 leveldb 用于查找键值设计的类

现在我们再看看 leveldb 具体是怎么实现的，`leveldb::MemTable::Get`实现如下：

```cpp
bool MemTable::Get(const LookupKey& key, std::string* value, Status* s) {
  Slice memkey = key.memtable_key();
  Table::Iterator iter(&table_);
  iter.Seek(memkey.data());
  if (iter.Valid()) {
    const char* entry = iter.key();
    uint32_t key_length;
    const char* key_ptr = GetVarint32Ptr(entry, entry + 5, &key_length);
    if (comparator_.comparator.user_comparator()->Compare(
            Slice(key_ptr, key_length - 8), key.user_key()) == 0) {
      // Correct user key
      const uint64_t tag = DecodeFixed64(key_ptr + key_length - 8);
      switch (static_cast<ValueType>(tag & 0xff)) {
        case kTypeValue: {
          Slice v = GetLengthPrefixedSlice(key_ptr + key_length);
          value->assign(v.data(), v.size());
          return true;
        }
        case kTypeDeletion:
          *s = Status::NotFound(Slice());
          return true;
      }
    }
  }
  return false;
}
```

`leveldb::MemTable::Get`通过`LookupKey` 构造出的`SkiplistKey`满足：

1. `UserKey`与需要查找的`UserKey`相同
2. `SequenceNumber`大于等于需要查找的`UserKey`

`MemTable`就可以查找到UserKey 可能出现的位置。leveldb 通过自定义的比较器（后文会详细分析比较器是如何实现的）将`MemTable`查找到的可能键值与参数传入搜索的键值对比，假如两个键值相同会再次判断`ValueType`。如果类型为`kTypeDeletion`说明该`UserKey`已经被删除，如果是`kTypeValue`那么直接返回最新版本的值。

至此我们就基本了解了 leveldb 在内存中做了什么，他通过组合 KeyValue 和封装`leveldb::Skiplist`实现了在只能存储 Key 并且 Insert-Only 的跳表上进行增删改查的功能。

## Imumtable MemTable
前文我们提到，当 `MemTable` 中积累一定数据就会把这部分数据转为 `Immutable MemTable`，并且后台线程会将`Immutable MemTable`持久化到磁盘变成 `SSTable`。那这块操作是如何实现的呢？

在`WAL`部分，我们讲过读写操作的底层实现，其中`MakeRoomForWrite`这个函数在每次写入操作之前被调用，用于确保有足够的空间来容纳新的写入。他的流程如下：

1. 检查当前 `MemTable` 的大小是否超过阈值。
2. 如果超过阈值，并且没有其他 `Immutable MemTable` 正在刷新，则将当前 `MemTable` 标记为 `Immutable`。
3. 创建一个新的 `MemTable` 来处理后续的写入操作。

```cpp
Status DBImpl::MakeRoomForWrite(bool force) {
  mutex_.AssertHeld();
  assert(!writers_.empty());
  bool allow_delay = !force;
  Status s;
  while (true) {
    if (!bg_error_.ok()) {
      // 如果后台有错误，直接返回
      s = bg_error_;
      break;
    } else if (allow_delay && versions_->NumLevelFiles(0) >= config::kL0_SlowdownWritesTrigger) {
      // 检查 Level-0 的文件数是否达到缓写阈值，
      // 如果达到阈值，则延迟 1 毫秒写入。
      // 通过延迟写入，降低写入速度，给后台Compaction线程腾出资源。
      mutex_.Unlock();
      env_->SleepForMicroseconds(1000);
      allow_delay = false;  // 最多延迟一次
      mutex_.Lock();
    } else if (!force && (mem_->ApproximateMemoryUsage() <= options_.write_buffer_size)) {
      // MemTable 大小未超过阈值，继续写入
      break;
    } else if (imm_ != nullptr) {
      // 已经有 Immutable MemTable 在刷新，等待完成
      bg_cv_.Wait();
    } else if (versions_->NumLevelFiles(0) >= config::kL0_StopWritesTrigger) {
      // L0 文件过多，停止写入
      bg_cv_.Wait();
    } else {
      // 将当前 MemTable 标记为 Immutable
      imm_ = mem_;
      has_imm_.store(true, std::memory_order_release);
      // 创建一个新的 MemTable
      mem_ = new MemTable(internal_comparator_);
      mem_->Ref();
      // 触发后台 Compaction
      MaybeScheduleCompaction();
      break;
    }
  }
  return s;
}
```

可以看到， 在 LevelDB 中`imm_`是 全局唯一 的。也就是说，在任意时刻，LevelDB 中最多只有一个 `MemTable` 会被标记为 `Immutable MemTable`，等待刷新到磁盘。当`MemTable`写满时，会创建一个新的`MemTable`，并且将旧的`MemTable`转换为`Immutable MemTable`

## 持久化 MemTable
持久化 `MemTable`的过程也称为 Minor Compaction，`MaybeScheduleCompaction`方法会触发后台的线程，通过`DBImpl::CompactMemTable`方法将数据刷新到磁盘。

```cpp
void DBImpl::CompactMemTable() {
  mutex_.AssertHeld();
  assert(imm_ != nullptr);

  // Save the contents of the memtable as a new Table
  VersionEdit edit;
  Version* base = versions_->current();
  base->Ref();
  Status s = WriteLevel0Table(imm_, &edit, base);
  base->Unref();

  if (s.ok() && shutting_down_.load(std::memory_order_acquire)) {
    s = Status::IOError("Deleting DB during memtable compaction");
  }     

  // Replace immutable memtable with the generated Table
  if (s.ok()) {
    edit.SetPrevLogNumber(0);
    edit.SetLogNumber(logfile_number_);  // Earlier logs no longer needed
    s = versions_->LogAndApply(&edit, &mutex_);
  }

  if (s.ok()) {
    // Commit to the new state
    imm_->Unref();
    imm_ = nullptr;
    has_imm_.store(false, std::memory_order_release);
    RemoveObsoleteFiles();
  } else {
    RecordBackgroundError(s);
  }
}
```

> 函数逻辑
>

1. 前置检查：确保当前线程持有锁， 并且确保存在待持久化的`Immutable MemTable`。
2. 将`immutable memtable`写入`sstable`：`WriteLevel0Table`函数后面会细说。为什么要通过`Ref`方法增加索引计数呢？我们设想一个情况：线程 A 正在持久化`imm_`，线程 B 调用`AppendVersion`导致当前`Version`的引用计数减一。若引用计数减为 0，那么当前`Version`被释放，线程 A 继续使用已经被释放的 `Version`，导致未定义行为。
3. 更新版本信息：设置前一个日志文件的编号为 0，表示旧的 WAL 文件不再需要。设置当前日志文件的编号，表示新的写入操作会记录到新的 WAL 文件中。
4. 提交新状态：将`imm_`置为空，删除不再需要的旧文件（如过期的 `WAL` 文件和 `SSTable` 文件）。

现在来看看`WriteLevel0Table`干了什么：

```cpp
Status DBImpl::WriteLevel0Table(MemTable* mem, VersionEdit* edit,
                                Version* base) {
  mutex_.AssertHeld();
  const uint64_t start_micros = env_->NowMicros();
  FileMetaData meta;
  meta.number = versions_->NewFileNumber();
  pending_outputs_.insert(meta.number);
  Iterator* iter = mem->NewIterator();
  Log(options_.info_log, "Level-0 table #%llu: started",
      (unsigned long long)meta.number);

  Status s;
  {
    mutex_.Unlock();
    s = BuildTable(dbname_, env_, options_, table_cache_, iter, &meta);
    mutex_.Lock();
  }

  Log(options_.info_log, "Level-0 table #%llu: %lld bytes %s",
      (unsigned long long)meta.number, (unsigned long long)meta.file_size,
      s.ToString().c_str());
  delete iter;
  pending_outputs_.erase(meta.number);

  // Note that if file_size is zero, the file has been deleted and
  // should not be added to the manifest.
  int level = 0;
  if (s.ok() && meta.file_size > 0) {
    const Slice min_user_key = meta.smallest.user_key();
    const Slice max_user_key = meta.largest.user_key();
    if (base != nullptr) {
      level = base->PickLevelForMemTableOutput(min_user_key, max_user_key);
    }
    edit->AddFile(level, meta.number, meta.file_size, meta.smallest,
                  meta.largest);
  }

  CompactionStats stats;
  stats.micros = env_->NowMicros() - start_micros;
  stats.bytes_written = meta.file_size;
  stats_[level].Add(stats);
  return s;
}
```

> 函数逻辑
>

1. 函数先创建了一个`FileMetaData`记录了`SSTable`文件的元数据。
2. 把新`SSTable`的编号记录到`pending_outputs_`中，告诉其他线程这个`SSTable`正在被构建中，不要把它误删除了。
3. 创建传入的`Immutable MemTable`的迭代器，通过`BuildTable`创建`SSTable`文件。`BuildTable`方法很简单，就是通过`WritableFile`接口，将迭代器的 kv 键值对写入`SSTable`文件。
4. 如果写入成功且文件非空，需要确定新文件应放在哪个层级：默认是 Level-0，但可能通过 `PickLevelForMemTableOutput` 选择更高层级。为什么需要 `PickLevelForMemTableOutput`？Level-0 的 `SSTable` 允许键范围重叠，但更高层级（Level-1 及以上）要求键范围严格有序且不重叠。若新 `SSTable` 的键范围与更高层级的文件无重叠，可直接将其放到更高层级，减少未来合并的次数。
5. 将文件信息记录到 `VersionEdit` 中，后续通过 `VersionSet` 提交生效。

现在回到`CompactMemTable`方法，我们将`edit`通过引用传递获得了新增`SSTable`的信息，将其应用在当前`Version`上，生成的新的`Version`并且插入`VersionSet`链表。

```cpp
 if (s.ok()) {
    edit.SetPrevLogNumber(0);
    edit.SetLogNumber(logfile_number_);
    s = versions_->LogAndApply(&edit, &mutex_);
  }
```

最后`RemoveObsoleteFiles`方法删除了磁盘上所有不需要的垃圾文件：

```cpp
// 删除磁盘上不再需要的文件，以释放磁盘空间。
// 随着时间的推移，可能会生成大量的临时文件或者过时的版本文件，
// 这些文件如果不及时删除，可能会占用大量的磁盘空间。
void DBImpl::RemoveObsoleteFiles() {
    mutex_.AssertHeld();

    if (!bg_error_.ok()) {
        // 如果存在后台错误，那么可能无法确定是否有新的版本提交，
        // 因此不能安全地进行垃圾收集，终止该次垃圾收集。
        return;
    }

    // 将所有需要用到的 SST 文件编号都记录到 live 中。
    //   - pending_outputs_: 正在进行 compaction 的 SST
    //   - versions_->AddLiveFiles(&live): 所有 version 里的 SST 
    std::set<uint64_t> live = pending_outputs_;
    versions_->AddLiveFiles(&live);

    // 获取 leveldb 目录下的所有文件名
    std::vector<std::string> filenames;
    env_->GetChildren(dbname_, &filenames);  // Ignoring errors on purpose
    uint64_t number;
    FileType type;

    // 遍历 leveldb 目录下的所有文件，
    // 把不再需要的文件记录到 files_to_delete 中。
    std::vector<std::string> files_to_delete;
    for (std::string& filename : filenames) {
        // 对于每个文件名，都调用 ParseFileName 解析出文件编号和文件类型。
        if (ParseFileName(filename, &number, &type)) {
            bool keep = true;
            switch (type) {
                case kLogFile:
                    // number >= versions_->LogNumber()，
                    // 表示这个 WAL 是最新或者未来可能需要的 WAL，
                    // 需要保留。
                    // number == versions_->PrevLogNumber()，
                    // 表示这个日志文件是上一个日志文件，
                    // 可能包含了一些还未被合并到 SST 文件的数据，也需要保留。
                    keep = ((number >= versions_->LogNumber()) ||
                            (number == versions_->PrevLogNumber()));
                    break;
                case kDescriptorFile:
                    // number >= versions_->ManifestFileNumber()，
                    // 表示这个 MANIFEST 文件是最新或者未来可能需要的 MANIFEST 文件，
                    // 需要保留。
                    keep = (number >= versions_->ManifestFileNumber());
                    break;
                case kTableFile:
                    // 之前已经将所需要的 SST 文件编号都记录到 live 中了，
                    // 如果当前 SST 文件编号在 live 中不存在，就表示不再需要了。
                    keep = (live.find(number) != live.end());
                    break;
                case kTempFile:
                    // 临时文件指正在进行 compaction 的 SST 文件，之前也已经提前
                    // 记录到 live 中了。如果当前临时文件不存在 live 中，表示不再需要了。
                    keep = (live.find(number) != live.end());
                    break;
                case kCurrentFile:
                    // CURRENT 文件，需要一直保留。
                case kDBLockFile:
                    // 文件锁，用来防止多个进程同时打开同一个数据库的，需要一直保留。
                case kInfoLogFile:
                    // INFO 日志文件，需要一直保留。
                    keep = true;
                    break;
            }

            if (!keep) {
                // 如果当前文件不需要保留了，将它加入到 files_to_delete 中，
                // 后面再一起删除。
                files_to_delete.push_back(std::move(filename));
                // 如果被删除的文件是个 SST，还需要把它从 table_cache_ 中移除。
                if (type == kTableFile) {
                    table_cache_->Evict(number);
                }
                Log(options_.info_log, "Delete type=%d #%lld\n", static_cast<int>(type),
                    static_cast<unsigned long long>(number));
            }
        }
    }

    // 这些需要被删除的文件，已经不会被访问到了。
    // 所以在删除期间，可以先释放锁，让其他线程能够继续执行。
    mutex_.Unlock();
    for (const std::string& filename : files_to_delete) {
        env_->RemoveFile(dbname_ + "/" + filename);
    }
    mutex_.Lock();
}
```

我们刚刚只是宏观的介绍了`immutable MemTable`是如何 dump 成为`SSTable`的，下一章节会具体讲解`SSTable`，关于版本控制的内容也会在后面细说。

## SSTable
### SSTable 结构
![](https://cdn.nlark.com/yuque/0/2025/svg/49907638/1738655463402-3357a5ca-c0fb-4190-8afd-96fdecbe99e0.svg)

`SSTable`中的数据按照功能可以分为如下几块区：

1. `Data Block`区：存放key/value数据。
2. `Meta Block`区：存放过滤器或当前SSTable相关的统计数据。
3. `MetaIndex Block`：仅有1个Block，该Block中存放了所有`Meta Block`的索引。
4. `Index Block`区：所有Data Block的索引（实际上只有一个`Index Block`）。
5. `Footer`：大小固定的一个区域（48B）。

LevelDB 通过`table_builder`对象完成`SSTable`的创建，但是我们先从`Footer`讲起。

### Footer
`Footer`包含四个部分：`MetaIndexHandle`，`IndexBlockHandle`，`Padding`，`MagicNum`。`MetaIndexHandle`指向`MetaIndex Block`，`IndexBlockHandle`指向`Index Block`，`MagicNum` 是 `SSTable`文件的标识符，帮助验证文件的完整性 。由于`Footer`固定 48B，因此需要用`Padding`填充。

因此当我们需要恢复/读取一个 `SSTable` 文件时，首先会读取文件的`Footer`得到`MetaIndex Block`和`Index Block`的位置，再通过他们两个知道`Data Blocks`和`Meta Blocks`的位置（偏移量），这时候既可以遍历数据了。

```cpp
class Footer {
 public:
  enum { kEncodedLength = 2 * BlockHandle::kMaxEncodedLength + 8 };
  Footer() = default;
  const BlockHandle& metaindex_handle() const { return metaindex_handle_; }
  void set_metaindex_handle(const BlockHandle& h) { metaindex_handle_ = h; }
  const BlockHandle& index_handle() const { return index_handle_; }
  void set_index_handle(const BlockHandle& h) { index_handle_ = h; }
  void EncodeTo(std::string* dst) const;
  Status DecodeFrom(Slice* input);
 private:
  BlockHandle metaindex_handle_;
  BlockHandle index_handle_;
};
```

`Footer`类内包含 `getter/setter`、`EncodeTo`、`DecodeFrom`四个函数，还有 48B 长度和两个 Handle，`MagicNumber`是定义在类外的静态常量。

`Footer`的写入位于`table/table_builder.cc`文件的`TableBuilder::Finish`函数中：

```cpp
// Write footer
  if (ok()) {
    Footer footer;
    footer.set_metaindex_handle(metaindex_block_handle);
    footer.set_index_handle(index_block_handle);
    std::string footer_encoding;
    footer.EncodeTo(&footer_encoding);
    r->status = r->file->Append(footer_encoding);
    if (r->status.ok()) {
      r->offset += footer_encoding.size();
    }
```

`Footer`的写入位于`TableBuilder::Finish`函数的末尾，这是因为它需要等待`Data Block`，`Meta Blocks`等写入后再写入。他实例化了一个`Footer`对象，然后传入索引的位置（偏移量），通过引用传递序列化得到`footer_encoding`最后写入文件。

`Footer`的读操作位于`table/table.cc`文件中的`Table::Open`函数中：

```cpp
Status Table::Open(const Options& options, RandomAccessFile* file,
                   uint64_t size, Table** table) {
  *table = nullptr;
  if (size < Footer::kEncodedLength) {
    return Status::Corruption("file is too short to be an sstable");
  }

  char footer_space[Footer::kEncodedLength];
  Slice footer_input;
  Status s = file->Read(size - Footer::kEncodedLength, Footer::kEncodedLength,
                        &footer_input, footer_space);
  if (!s.ok()) return s;

  Footer footer;
  s = footer.DecodeFrom(&footer_input);
  if (!s.ok()) return s;

```

读操作就是写操作的逆过程，我们需要根据`Footer`中存储的信息反向解析一整个`SSTable`。因为`Footer`位于文件末尾，将文件大小减去`Footer`大小就可以得到它的内容，再将它反序列化得到一个`Footer`实例。`DecodeFrom`方法就是前面的代码实现：![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738746863737-4cd4c7ce-8bfb-4434-86b8-c30f59c47988.png)

### Data Blocks
`SSTable`中各个 Block 都以下图格式组织：	![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738778092999-fcd0e030-502b-4d06-8928-079cd5d17277.png)![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738835486944-42572eb7-dd64-4dbc-baaf-d038539ca8d3.png)

<font style="color:rgb(25, 27, 31);">每个 Block 可分为三个部分，分别是

+ `<font style="color:rgb(25, 27, 31);">Entry`
+ `<font style="color:rgb(25, 27, 31);">Restart Point`
+ `<font style="color:rgb(25, 27, 31);">Restart Point Count`

<font style="color:rgb(25, 27, 31);">在回答什么是 `<font style="color:rgb(25, 27, 31);">Restart Point`<font style="color:rgb(25, 27, 31);"> 之前，我们先来想一个问题。我们知道 LevelDB 存储`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">的数据是有序排列的，那么如果通过顺序遍历查找的效率就相对较慢了，还有没有别的办法呢？对，数据结构中分块查找的思路就很适合这种情况。![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738778543797-347ccc29-b756-4db2-aa93-1ea00be46b61.png)

我们可以指定一些 KV 键值对（`Entry`）为索引（重启点 `Restart Point`），这样就<font style="color:rgba(0, 0, 0, 0.87);">能够迅速判断目标键可能位于哪个数据块范围，跳过大量明显不包含目标键的数据块，无需对整个SSTable进行顺序扫描（这其实和跳表也是一个思路）。

`<font style="color:rgba(0, 0, 0, 0.87);">Entry`<font style="color:rgba(0, 0, 0, 0.87);">就是对 KV 键值对的序列化，每个Entry可分为5段，分别为：该`Entry`的 Key 与其相应的`Restart Point`的公共前缀长度（Varint32编码）、该`Entry`的 Key 剩余的长度（Varint32编码）、该`Entry`的 Value 长度（Varint32）编码、该`Entry`的 Key的 非公共前缀数据（bytes）、该`Entry`的 Value 数据（bytes）。

**我们需要注意区分一下这里的**`**Entry**`**和前文在**`**SkipList**`**中提到的**`**SkipListKey**`**。**`SkipList`中只能存储 Value，所以我们将 KV 键值对序列化为`SkipListKey`存储在跳表中。我们知道 `SkipList`也就是`MemTable`会被 dump 到磁盘变成`SSTable`。此时 LevelDB 会利用`MemTable`的 Iterator 遍历每一个 KV 键值对（LevelDB 会反序列化`SkipListKey`），再生成`Entry`存储到`SSTable`，这里是为了利用它有序的特性压缩存储。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738826716290-dce30310-c948-4438-a8c6-3a8741c09a41.png)	由于`SSTable`存储的数据具有有序排列的性质，我们利用`Restart Point`来优化查找，那么有没有办法优化存储减少占用空间呢？

我们设想一个场景，现在我们需要 LevelDB 存储每天的开销：

```plain
May-01  -->  100
May-02  -->  80
May-03  -->  10
May-04  -->  900
May-05  -->  700
```

存储的 Key 都是以 May 开头，这是我们只需要`Restart Point`存储下完整的 Shared_Key，其余键值对只需要保存 Non_Shared_Key 例如-01、-02、-03 等，以此节省空间。

> <font style="color:rgb(64, 64, 64);">LevelDB 设计 Restart Point 的目的是在读取`<font style="color:rgb(64, 64, 64);">SSTable`<font style="color:rgb(64, 64, 64);">内容时，加速查找的过程。
>
> <font style="color:rgb(64, 64, 64);">由于每个 Restart Point 存储的都是完整的 Key 值，因此在 `<font style="color:rgb(64, 64, 64);">SSTable`<font style="color:rgb(64, 64, 64);"> 中进行数据查找时，可以首先利用 Restart Point 点的数据进行键值比较，以便于快速定位目标数据所在的区域；
>
> <font style="color:rgb(64, 64, 64);">当确定目标数据所在区域时，再依次对区间内所有数据项逐项比较 Key 值，进行细粒度地查找；
>
> <font style="color:rgb(64, 64, 64);">该思想有点类似于跳表中利用高层数据迅速定位，底层数据详细查找的理念，降低查找的复杂度。
>

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738830937852-22f066a9-c939-4fc8-b55a-9ffa84f720bc.png)

> 可以注意到将数据写入`buffer_`时出现了两个方法`PutVarint32`和`append`，前者是用于写入可变长数字，后者是写入`std::string`，具体原因在下一章编码格式会提及。
>

代码实现如上，我们可以用上面的例子体会一下（假设`block_restart_interval=2`）：

```plain
May-01  -->  100       | 0 | 6 | 3 | "May-01" | 100 |   // Restart Point
May-02  -->  80        | 4 | 2 | 2 |   "01"   |  80 |          
May-03  -->  10        | 0 | 6 | 2 | "May-03" |  10 |   // Restart Point
May-04  -->  900       | 4 | 2 | 3 |   "04"   | 900 | 
May-05  -->  700       | 0 | 6 | 3 | "May-05" | 700 |   // Restart Point
```

写入完`Entry`部分数据就是`Restart Point`和`Restart Point Count`两个元信息。在`BlockBuilder::Finish `中我们可以更清晰地了解 Block 的结构：

```cpp
Slice BlockBuilder::Finish() {
    // Append restart array
    for (size_t i = 0; i < restarts_.size(); i++) {
        PutFixed32(&buffer_, restarts_[i]);
    }
    PutFixed32(&buffer_, restarts_.size());
    finished_ = true;
    return Slice(buffer_);
}
```

<font style="color:rgb(25, 27, 31);">LevelDB 实现了 `<font style="color:rgb(25, 27, 31);">BlockIterator`<font style="color:rgb(25, 27, 31);"> 用来从 Block 中查找数据，由于 Block 中的数据是有序的，所以使用[<font style="color:rgb(25, 27, 31);">二分法](https://zhida.zhihu.com/search?content_id=235276576&content_type=Article&match_order=1&q=%E4%BA%8C%E5%88%86%E6%B3%95&zhida_source=entity)<font style="color:rgb(25, 27, 31);">即可。首先在重启点 `<font style="color:rgb(25, 27, 31);">Resrart Point`<font style="color:rgb(25, 27, 31);"> 之间进行二分查找，找到所在区域时再每个`<font style="color:rgb(25, 27, 31);">Entry`<font style="color:rgb(25, 27, 31);">依次遍历。 

### <font style="color:rgb(25, 27, 31);">Index Block
了解了 Data Blocks，那么 Index Block 也很简单了。我们知道每个 Block 都是由`Entry`，`Restart Point`和`Restart Point Count`组成的。对于 Data Blocks，他的`Entry`存储的是`MemTable`中的键值对；对于 Index Block，`Entry`存储的就是每个 Data Block 的索引（偏移量）。Key 值就是大等于对应 Data Block 的最小的 Key，Value 值就是Data Block 偏移 + Data Block 大小。在 Index Block 中每一个 `Entry` 都是重启点，也就是<font style="color:rgb(25, 27, 31);">稠密索引。我们查找一个数据时，首先在 Index Block 中进行二分查找，知道目标 Key 在哪一个 Data Block，之后再到 Data Block 中查找。

<font style="color:rgb(25, 27, 31);">LevelDB 将 `<font style="color:rgb(25, 27, 31);">MemTable`<font style="color:rgb(25, 27, 31);">生成`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">的操作封装到了`<font style="color:rgb(25, 27, 31);">TableBuilder`<font style="color:rgb(25, 27, 31);">这个类中，而`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">中包含多个 Block，LevelDB 封装了`<font style="color:rgb(25, 27, 31);">BlockBuilder`<font style="color:rgb(25, 27, 31);">类操作这些 Block。

<font style="color:rgb(25, 27, 31);">我们先看看`<font style="color:rgb(25, 27, 31);">TableBuilder`<font style="color:rgb(25, 27, 31);">是如何使用的：

```cpp
// 创建SST文件
WritableFile* file;
s = env->NewWritableFile(fname, &file);
if (!s.ok()) {
    return s;
}

// 创建一个TableBuilder对象，
// 用于将MemTable中的数据写入到SST文件中
TableBuilder* builder = new TableBuilder(options, file);

// 通过TableBuilder对象将
// 所有kv先写入到各个`Block`缓冲区里。
Slice key;
for (; iter->Valid(); iter->Next()) {
    key = iter->key();
    builder->Add(key, iter->value());
}

// 将各个`Block`缓冲区里的内容写入到文件中，
// 该`SST`文件构建完成。
builder->Finish();
delete builder;
```

当我们将`MemTable`中的`SkipListKey`插入`SSTable`时，首先会调用`TableBuilder::Add`，之后会再调用`BlockBuilder::Add`将 kv 键值对插入 Data Blocks 中。![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738912186885-924abe88-4644-4709-8e1f-3600849e5f4f.png)

在研究`Flush`方法之前，我们先看一个细节。之前我们提过，Index Block 中存储的 Key 是大等于对应 Data Block 的下一个键值，那为什么我们不直接用对应 Data Block 的最后一个 Key 呢？我们再来设想一个情况：上一个 Data Block 最后一个 Key 是 aaaabccccc，下一个 Data Block 的第一个 Key 是 aaaabddddd。对于 Index Block 我们需要的 Key 只求能够区分两个 Data Block，所以 aaaabccccc 和 aaaabd 他们的效果是相同的，我们需要一个 Key 满足 Last_Key <= Key < Current_Key，因此采用了`FindShortestSeparator`方法。

现在我们再看看`Flush`方法，它的作用就是结束当前 Data Block 的创建（调用`WriteBlock`写入磁盘），并且为下一个 Data Block 的创建做准备。

```cpp
void TableBuilder::Flush() {
  Rep* r = rep_;
  assert(!r->closed);
  if (!ok()) return;
  if (r->data_block.empty()) return;
  assert(!r->pending_index_entry);
  WriteBlock(&r->data_block, &r->pending_handle);
  if (ok()) {
    r->pending_index_entry = true;
    r->status = r->file->Flush();
  }
  if (r->filter_block != nullptr) {
    r->filter_block->StartBlock(r->offset);
  }
}
```

可以看到每次`TableBuilder::Add`判断 Data Block 填满时，都会将`pending_index_entry`置为`true`，下一次`TableBuilder::Add`就会在 Index Block 添加。`WriteBlock`将 Data Block 压缩后写入到 `SSTable` 文件中，并生成 Block Handle。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738916975629-9ebd2847-b698-48e6-bd79-40d5505a070e.png)

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738918405090-3814b205-fe05-460a-b340-2cbe24b21164.png)

至此，离了解 Index Block 还差最后一块：`TableBuilder::Finish`。

```cpp
// Write index block
  if (ok()) {
    if (r->pending_index_entry) {
      r->options.comparator->FindShortSuccessor(&r->last_key);
      std::string handle_encoding;
      r->pending_handle.EncodeTo(&handle_encoding);
      r->index_block.Add(r->last_key, Slice(handle_encoding));
      r->pending_index_entry = false;
    }
    WriteBlock(&r->index_block, &index_block_handle);
  }
```

这块代码你是不是很熟悉？和`TableBuilder::Add`几乎一模一样，那为什么还需要再写一遍呢？因为每次调用`TableBuilder::Add`，当 Data Block 写满时才会写入磁盘。所以你写入最后一条 kv 键值对时不一定会调用`TableBuilder::Flush`。

### Meta Blocks(Filter Block)
Meta Blocks 用来存储一些元信息，由于目前的 LevelDB 中仅存储了布隆过滤器，所以也可以叫做 Filter Block。如果打开数据库时指定了”FilterPolicy”, 那么每个 Table 中都会存储一个 Filter Block，每个 Filter Block 包含多个 Filter。

<font style="color:rgb(25, 27, 31);">布隆过滤器是一种[<font style="color:rgb(25, 27, 31);">数据结构](https://zhida.zhihu.com/search?content_id=143205931&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84&zhida_source=entity)<font style="color:rgb(25, 27, 31);">，一种巧妙的概率型数据结，特点是高效地插入和查询，可以用来告诉你某个键一定不存在或者可能存在。相比`Map`和`Set`布隆过<font style="color:rgb(25, 27, 31);">滤器占用的空间少很多，但是结果具有假阳性，如果返回键不存在，那么键一定不存在，如果返回键存在，那么键有可能不存在、又有可能存在。

<font style="color:rgb(25, 27, 31);">那布隆过滤器是如何帮助我们提高查找效率的呢？我们回忆一下在`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">中读取一个 Key 的过程：

1. 读取`SSTable`的`Footer`，根据里面的信息再读取 Index Block 与 Meta Index Block。
2. 根据 Meta Index Block 读取布隆过滤器到内存。
3. 在内存中对 Index Block 进行二分查找，得到 Key 所在的 BlockHandle（也就是偏移）。
4. 根据 BlockHandle 获取布隆过滤器的编号，也就是偏移。
5. 通过布隆过滤器判断 Key 是否存在，不存在则结束。（这里节省了时间开销）
6. 如果存在，那么读取对应的 Data Block，对 `Restart Point`进行二分查找。
7. 最后在对应`Restart Point`进行遍历。

接下来再看看 Filter Block 的格式：

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1738995368007-efd7e32b-a5f1-4a20-b0f9-71dba91b7149.png)

+ `base_lg` 是每个布隆过滤器大小（默认是 11），表示每隔 2KB << 11 的数据开启一个布隆过滤器。
+ `filter data`是每个布隆过滤器的内容。
+ `filter data offsets`是一个数组，记录了每个`filter data`的偏移量。
+ `filter data size`标记了`filter data offsets`的位置。

`filter data size`固定占用 4B，`base_lg`固定占用 1B，从而我们可以反向读取 Filter Block 获取每一个不定长部分的数据。

值得注意的是，Filter 和 Data Block 不是一一对应的，多个 Data Block 可能共用一个 Filter。这是没问题的：假如Data_Block_M、Data_Block_M+1、Data_Block_M+共用一个 Filter，现在来判定 Key 有没有可能存在于 Data_Block_M+1。若结果为false，那么 Key 不可能存在于这3个 Data Block 中的任何一个。所以 Filter 正确性是保证的。然而，它增大了false-positive的可能性。为此，需要控制共用的范围，大约2KB数据共用一个filter：

```cpp
// Generate new filter every 2KB of data
static const size_t kFilterBaseLg = 11;
static const size_t kFilterBase = 1 << kFilterBaseLg;
```

现在再看看 Filter Block 的代码实现，`FilterBlockBuilder`负责在`SSTable`中构建 Filter Block，`FilterBlockReader`负责解析读取 Filter Block。我们通过 Filter Block 的写入流程具体分析：

1. 搜集传进来的 Key。调用`TableBuilder::AddKey`插入 `SkipListKey`时，会调用`FilterBlockBuilder::AddKey`记录下 Key。

```cpp
void FilterBlockBuilder::AddKey(const Slice& key) {
  Slice k = key;
  start_.push_back(keys_.size());
  keys_.append(k.data(), k.size());
}
```

> 这里可以注意到 LevelDB 保存 Key 是将所有 Key 存在一个`std::string`中，并且记录下每一个 Key 的位置。为什么不用`vector<std::string>`直接保存呢？可能是出于内存使用效率和性能优化的考虑，使用单个 `std::string`存储所有 Key 可以避免大量小字符串带来的内存碎片问题。如果使用`vector<std::string>`，每个`std::string`会有独立的内存分配。
>

2. 每当一个 Data Block 写入完毕（也就是前文提到的`TableBuilder::Flush`）或者`TableBuilder`实例化时，`FilterBlockBuilder::StartBlock`就会被调用：

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739090502229-d1af7e4b-cf53-430c-a8b4-9de99f48ad28.png)

`StartBlock`方法是为了保存上一个 Data Block 对应 Key，并且生成所有的 Filter（LevelDB 默认每 2KB 的 Data Block 数据生成一个 Filter）。

```cpp
void FilterBlockBuilder::StartBlock(uint64_t block_offset) {
  uint64_t filter_index = (block_offset / kFilterBase);
  assert(filter_index >= filter_offsets_.size());
  while (filter_index > filter_offsets_.size()) {
    GenerateFilter();
  }
}
```

`FilterBlockBuilder::StartBlock`理解花费了我很长时间，我们先来看看 LevelDB 在判断 Key 是否位于一个 Data Block 时候是怎么做的。

LevelDB 在查询一个 Key 的时候会调用`FilterBlockReader::KeyMayMatch`快速判断某个键是否可能存在于指定数据块中。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739092764015-489a4961-d6b6-4b63-ac43-faec16ec0396.png)

这时候我们就能很轻松的理解，为什么`FilterBlockBuilder::StartBlock`要计算`filter_index`（不理解的可以结合代码和 Filter Block 结构图再看看）。`StartBlock`每次计算出当前 Data Block 所对应 Filter 的序号，之后用`start_`中保存的 Key 生成 Filter。可是为什么要判断 `filter_index > filter_offsets.size()`呢？因为前面说过 **Data Block 和 Filter 不是一一对应的**。Filter 与 Data Block 的对应关系是基于数据偏移量 2KB 的连续划分，这说明一个 Filter 可能对应多个 Data Block 或者多个 Filter 对应一个 Data Block，取决于设置的 Data Block 大小。![](https://cdn.nlark.com/yuque/0/2025/jpeg/49907638/1739094104848-03f64dee-806f-4872-9732-200635408e24.jpeg)

其实这幅图还有一个地方有问题。例如 Data Block 大小为 4KB，Filter 每 2KB 记录一次。那么会生成两个 Filter，`filter_offsets_`数组会记录 filter-0 的偏移，而不是图中的 empty。我们看看代码：![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739095476860-528ccac1-4823-414e-b4e9-e8d30977c379.png)![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739096101281-27335166-89b4-42fa-a095-30a3ca3d8431.png)

LevelDB 的布隆过滤器具体是怎么实现的后面会开一个专题细说。

### Meta Index Block
 Meta Index Block 与 Index Block 相似。Meta Index Block 记录 Filter Block 在 `SSTable`中的位置。还记得`TableBuilder::WriteRawBlock`这个方法吗？前面 Filter Block 写入 `SSTable`文件就是通过这个方法。除此之外，LevelDB 将`handle`作为参数传入，得到了 Filter Block 的偏移和大小信息。

```cpp
void TableBuilder::WriteRawBlock(const Slice& block_contents,
                                 CompressionType type, BlockHandle* handle) {
  Rep* r = rep_;
  handle->set_offset(r->offset);
  handle->set_size(block_contents.size());
}
```

之后 `TableBuilder::Finish`通过`handle`组合了 kv 键值对（Key 是 filter.bloom，Value 是 Filter Block 的偏移和大小），插入`metaindex_block_handle`这个对象中，最后写入`SSTable`。

> `BlockHandle`这个类是 LevelDB 设计出来管理 Block 信息的，它包括 Block 的偏移和大小。
>

```cpp
BlockHandle metaindex_block_handle;
// Write metaindex block
if (ok()) {
    BlockBuilder meta_index_block(&r->options);
    if (r->filter_block != nullptr) {
      // Add mapping from "filter.Name" to location of filter data
      std::string key = "filter.";
      key.append(r->options.filter_policy->Name());
      std::string handle_encoding;
      filter_block_handle.EncodeTo(&handle_encoding);
      meta_index_block.Add(key, handle_encoding);
    }
    WriteBlock(&meta_index_block, &metaindex_block_handle);
}
```

至此 LevelDB 的`SSTable`部分全部结束了。

## 编码格式
在介绍 SSTable Compaction 之前，我们先插入介绍一下 LevelDB 的编码方式。

LevelDB 对不同类型的数据采用不同的编码方式，大体上可以分为整数和字符串。LevelDB 对整数采用小端存储（Little Endian），对于不同情况采用定长编码或变长编码。对字符串采用长度[前缀编码](https://zhida.zhihu.com/search?content_id=136166411&content_type=Article&match_order=1&q=%E5%89%8D%E7%BC%80%E7%BC%96%E7%A0%81&zhida_source=entity)的方式存储。

### 定长整数
定长整数的小端存储很简单。例如 32 位整数，64 的二进制表示是 00000000,00000000，00000000，01000000，存储时需要将最低有效字节 01000000 放在低地址部分。

`FixedInt`编解码速度快，但是会浪费空间，属于空间换时间的做法。频繁调用并且出现值较大数的概率偏大时适合用`FixedInt`。

```cpp
inline void EncodeFixed32(char* dst,  value) {
  uint8_t* const buffer = reinterpret_cast<uint8_t*>(dst);

  buffer[0] = static_cast<uint8_t>(value);
  buffer[1] = static_cast<uint8_t>(value >> 8);
  buffer[2] = static_cast<uint8_t>(value >> 16);
  buffer[3] = static_cast<uint8_t>(value >> 24);
}

inline void EncodeFixed64(char* dst, uint64_t value) {
  uint8_t* const buffer = reinterpret_cast<uint8_t*>(dst);

  buffer[0] = static_cast<uint8_t>(value);
  buffer[1] = static_cast<uint8_t>(value >> 8);
  buffer[2] = static_cast<uint8_t>(value >> 16);
  buffer[3] = static_cast<uint8_t>(value >> 24);
  buffer[4] = static_cast<uint8_t>(value >> 32);
  buffer[5] = static_cast<uint8_t>(value >> 40);
  buffer[6] = static_cast<uint8_t>(value >> 48);
  buffer[7] = static_cast<uint8_t>(value >> 56);
}

```

`EncodeFixed`的思路就是通过移位操作将每 8bit 的 value 移入指针处内存。仔细研究一下它的代码实现：

> 为什么要将 `char*`转为`uint8_t*`
>

因为 `char` 在不同平台的表示范围不一样。有的平台是`unsigned char`，表示范围为 0~255； 有的平台是`signed char`，表示范围为-128~127。 为了确保我们处理的是无符号的字节，并且让代码更加明确和安全，通常会使用 标准库定义的 无符号字节类型 `uint8_t`（无符号的 8 位整数）来代替 `char` 类型 。

> 为什么`static_cast<uint8_t>(value >> k)`可以取出数据。
>

将 32 位整数转为 8 位整数会直接截取高位部分保留低位部分，`static_cast<uint8_t>(value)`就可以直接去除低 8 位赋值进 `buffer[0]`。右移运算会移出低位，`>> 8`后就可以取出 9~15 位，以此类推。

### 变长整数
对于较小的整型数据使用`FixedInt`容易浪费空间。例如上文中，利用`FixedInt`存储 32 位型 64 需要使用 4B 的内存，但实际上前 3B 存储的都是零扩展也就是无效数据。leveldb 采用`VarInt`来编码这些较小的整型数据。<font style="color:rgb(25, 27, 31);">它的原理就是只使用一个字节的低7位存储数据，而高位用来做标识，高位为1的时候表示需要继续读取下一个字节，高位为0的时候表示当前字节已是最后一个字节。存储也是采用Little Endian的方式。例如：十进制数字`<font style="color:rgb(25, 27, 31);">500`<font style="color:rgb(25, 27, 31);">的二进制表示为`<font style="color:rgb(25, 27, 31);">0001 1111 0100`<font style="color:rgb(25, 27, 31);">， 以`<font style="color:rgb(25, 27, 31);">VarInt32`<font style="color:rgb(25, 27, 31);">存储

+ <font style="color:rgb(25, 27, 31);">低字节 7bit `<font style="color:rgb(25, 27, 31);">1110100`<font style="color:rgb(25, 27, 31);"> -> `<font style="color:rgb(25, 27, 31);">1 1110100`<font style="color:rgb(25, 27, 31);">，后面还有数据，控制位为 1
+ <font style="color:rgb(25, 27, 31);">高字节 7bit `<font style="color:rgb(25, 27, 31);">0000011`<font style="color:rgb(25, 27, 31);"> -> `<font style="color:rgb(25, 27, 31);">0 0000011`

<font style="color:rgb(25, 27, 31);">所以这个例子中，十进制的`<font style="color:rgb(25, 27, 31);">500`<font style="color:rgb(25, 27, 31);">被编码为`<font style="color:rgb(25, 27, 31);">11110100 00000011`<font style="color:rgb(25, 27, 31);">占两字节。`<font style="color:rgb(25, 27, 31);">VarInt`<font style="color:rgb(77, 77, 77);">编解码速度慢，但是节省空间，属于时间换空间的做法。<font style="color:rgba(0, 0, 0, 0.75);">当数据量较大，且经常出现较小数值时，选择`Varint`<font style="color:rgba(0, 0, 0, 0.75);">可以节省存储空间，但不会损失多少性能。

`<font style="color:rgba(0, 0, 0, 0.75);">util/coding.h/leveldb::EncodeVarint32`<font style="color:rgba(0, 0, 0, 0.75);">的实现如下：

```cpp
char* EncodeVarint32(char* dst, uint32_t v) {
  // Operate on characters as unsigneds
  uint8_t* ptr = reinterpret_cast<uint8_t*>(dst);
  static const int B = 128;
  if (v < (1 << 7)) {
    *(ptr++) = v;
  } else if (v < (1 << 14)) {
    *(ptr++) = v | B;
    *(ptr++) = v >> 7;
  } else if (v < (1 << 21)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = v >> 14;
  } else if (v < (1 << 28)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = v >> 21;
  } else {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = (v >> 21) | B;
    *(ptr++) = v >> 28;
  }
  return reinterpret_cast<char*>(ptr);
}
```

上述代码的核心在于每 7bit 分割一次并且设置控制位。

> *(ptr++) = v; 就是将 v 的低 8 位直接赋值到指针处，并且让指针指向下一个字节位置。
>

为什么直接赋值低 8 位，不应该是每 7 位存储吗？因为`if(v < (1 << 7))`的条件限制了`v`小于 128，第 8 位一定是 0。当`v`大等于 128 时就需要分多个字节存储，`*(ptr++) = v|B`的目的就是取 0~7 位，并且将第 8 位置为 1，`*(ptr++)= v >> 7`就是将 8~15 位赋值到下一块内存（和上面一样 ifelse 限定了`v`小于 16384 也就是最高位一定为 0）。

`util/coding.h/leveldb::EncodeVarint64`定义如下：

```cpp
char* EncodeVarint64(char* dst, uint64_t v) {
  static const int B = 128;
  uint8_t* ptr = reinterpret_cast<uint8_t*>(dst);
  while (v >= B) {
    *(ptr++) = v | B;
    v >>= 7;
  }
  *(ptr++) = static_cast<uint8_t>(v);
  return reinterpret_cast<char*>(ptr);
}
```

`Varint32`与`Varint64`的逻辑是一样的，但是为什么`Varint32`不也用 while 循环呢？`EncodeVarint32`处理的是32位的整数，最多只需要5个字节就可以表示。因此，它可以通过分组复制的方式，直接将整数的每个部分编码到结果中。这种方式的代码比较直观，易于理解。而`EncodeVarint64`处理 64 位整数，最多需要 10 个字节。如果采用和  `VarInt32`一样的分组复制代码比较冗长，而 while 比较直观简洁，两种实现方式在效率上没有差别。

## Compaction
Compaction 是 LevelDB 最为复杂的过程之一，同样也是leveldb的性能瓶颈之一。在 LevelDB 中分为 Minor Compaction 和 Major Compaction 两种合并，Minor Compaction 是将内存中 `Immutable MemTable` 持久化到磁盘变成 `Level-0 SSTable`的过程，前面已经讲解过了；Major Compaction 是将下层 `SSTable`合并到上层的过程。

在数据更新这个部分我已经介绍了 LevelDB 插入数据的方法， LevelDB 插入/删除一条数据是通过插入一条新的记录，因此 LevelDB 会存在大量的<font style="color:rgb(34, 34, 34);">冗余数据。Major Compaction 不仅能够节省磁盘空间还能够提高读效率。

但是 Major Compaction 的过程其本质是一个多路归并的过程，既有大量的磁盘读开销，也有大量的磁盘写开销，显然这是一个严重的性能瓶颈。所以如何触发 Major Compaction？ 什么时候执行 Major Compaction？需要 Major Compaction 那些 `SSTable`？Major Compaction 具体怎么做？接下来我们就要讨论这些问题。

### 如何触发
LevelDB 的后台线程调度模型比较简单：

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739254956029-c3adb6ce-132d-4567-989f-cadb145649ff.png)

1. LevelDB 会在可能需要 Compation 的地方调用`DBImpl::MaybeScheduleCompaction`方法
2. `MaybeScheduleCompaction`方法会通过`Env`类<font style="color:rgb(25, 27, 31);">调用不同环境中的`<font style="color:rgb(25, 27, 31);">Schedlue`<font style="color:rgb(25, 27, 31);">方法。
3. `<font style="color:rgb(25, 27, 31);">env->Schedule`<font style="color:rgb(25, 27, 31);">方法会通过后台线程执行 Compation。

> Env 类在 LevelDB 中是一个抽象基类，它定义了一组虚拟方法，这些方法封装了所有与操作系统环境交互的操作。这包括文件操作（如打开、读取、写入、关闭文件），线程创建和同步操作（如互斥锁和条件变量），以及获取系统相关信息（如当前时间，或者某个文件的大小）等。这种设计使得 LevelDB 可以在不同的操作系统和平台上运行，只需要提供一个特定平台的 Env 实现。
>

```cpp
void DBImpl::MaybeScheduleCompaction() {
  mutex_.AssertHeld();
  if (background_compaction_scheduled_) {
    // Already scheduled
  } else if (shutting_down_.load(std::memory_order_acquire)) {
    // DB is being deleted; no more background compactions
  } else if (!bg_error_.ok()) {
    // Already got an error; no more changes
  } else if (imm_ == nullptr && manual_compaction_ == nullptr &&
             !versions_->NeedsCompaction()) {
    // No work to be done
  } else {
    background_compaction_scheduled_ = true;
    env_->Schedule(&DBImpl::BGWork, this);
  }
}
```

`MaybeScheduleCompaction`方法会先检查一些前置条件，最终调用`env->Schedule`。在<font style="color:rgb(37, 41, 51);">合并过程中都是加锁的，而且设置了`<font style="color:rgb(37, 41, 51);">background_compaction_scheduled_`<font style="color:rgb(37, 41, 51);">的判断，所以同一个时间只会有一个线程在合并。

```cpp
void WindowsEnv::Schedule(
    void (*background_work_function)(void* background_work_arg),
    void* background_work_arg) {
  background_work_mutex_.Lock();

  // Start the background thread, if we haven't done so already.
  if (!started_background_thread_) {
    started_background_thread_ = true;
    std::thread background_thread(WindowsEnv::BackgroundThreadEntryPoint, this);
    background_thread.detach();
  }

  // If the queue is empty, the background thread may be waiting for work.
  if (background_work_queue_.empty()) {
    background_work_cv_.Signal();
  }

  background_work_queue_.emplace(background_work_function, background_work_arg);
  background_work_mutex_.Unlock();
}
```

`WindowsEnv::Schedule`的执行逻辑和我预想的不同。我以为会每次通过`std::thread`打开一个新线程执行任务（对多线程了解不足），实际上 LevelDB 做的事情是用`std::thread`创建一个常驻线程（通过`detach`方法脱离父进程，这样父进程销毁了它还能运行）。它会循环获取任务队列中的任务，为了避免线程空转，在队列为空时通过信号量等待唤醒。如果队列中有任务，则获取该任务并将任务出队，然后执行任务。后台线程中操作队列的部分需要通过锁来保护，而执行任务时没有上锁，可以并行执行（但是LevelDB只使用了1个后台线程，因此Compaction仍是串行而不是并行的）。

接下来我们看看这个后台线程具体是怎么样的：

```cpp
void WindowsEnv::BackgroundThreadMain() {
  while (true) {
    background_work_mutex_.Lock();

    // Wait until there is work to be done.
    while (background_work_queue_.empty()) {
      background_work_cv_.Wait();
    }

    assert(!background_work_queue_.empty());
    auto background_work_function = background_work_queue_.front().function;
    void* background_work_arg = background_work_queue_.front().arg;
    background_work_queue_.pop();

    background_work_mutex_.Unlock();
    background_work_function(background_work_arg);
  }
}
```

`BackgroundThreadMain`在任务队列为空时，会主动阻塞`background_work_cv_`。等到下一次 LevelDB 通过`Schedule`执行任务时，任务队列为空此时就会解锁`background_work_cv_`唤醒进程，这种操作高效地节省了系统资源。

我们了解了 LevelDB 后台线程调度的机制，现在回到`DBImpl::MaybeScheduleCompaction`。这个方法会通过后台线程调用`BackgroundCompaction`，执行实际的 Compaction 工作。

### 触发时机
#### Manual Compaction
通过`DBImpl::TEST_CompactRange`触发，为了测试而存在。Manual Compaction 的信息通过`ManualCompaction`类进行管理。

```cpp
struct ManualCompaction {
    int level;
    bool done;
    const InternalKey* begin;  // null means beginning of key range
    const InternalKey* end;    // null means end of key range
    InternalKey tmp_storage;   // Used to keep track of compaction progress
};
```

`DBImpl::TEST_CompactRange`会构造一个 Manual Compaction，然后循环完成 Compaction。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739263968585-8113d52d-62ab-4da7-a726-76da2dee7aba.png)

为什么这里需要用 while 循环呢？首先在 Manual Compaction 的时候有可能正在进行 Compaction，需要等待后台线程完成。其次 Manual Compaction 可能需要多个阶段，这是啥什么意思呢？我们深入看看代码：

`void DBImpl::TEST_CompactRange(int level, const Slice* begin, const Slice* end);`需要传入`begin`和`end`。`begin`和`end`之间的文件可能非常多，为了保证 Compaction 的效率，不会一次 Compaction 完所有的文件，会先选择一个`middle`，然后 Compaction `begin`和`middle`之间的文件，完成后将`begin`设置成`middle`，将`DBImpl::manual_compaction_`设置为`null`，这样下次循环的时候会继续 Compaction `middle`和`end`之间的文件。当最后一次 Compaction 完成后，`done`设置为`true`，循环就退出了。

#### Size Compaction
Size Compaction 根据每层总 `SSTable` 大小触发（ Level-0 根据 `SSTable` 数量）。由于只有 Compaction 回改变`SSTable`大小，并且 Compaction 之后会进行 `Version`更新，所以 LevelDB 选择在`VersionSet::Finalize`函数中计算`compaction_score_`。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739268207942-2a677640-d325-4ac2-b4cc-df21bbabb41a.png)

计算后得到`compaction_score_`和`compaction_level_`，若得分大等于 1 那么就进行 Size Compactiom。

#### Seek Compaction
每个新的`SSTable`文件维护一个`allowed_seek`的初始阈值，表示最多容忍多少次 seek miss。当`allowed_seeks`递减到小于0的时候，就会将对应文件标记为需要 Compaction。

> <font style="color:rgb(25, 27, 31);">在搜索`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">时，会找到键范围包含待搜索键的`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">，从 Level-0 开始搜索，如果一个`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">没有找到，会搜索下一个`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">，直到找到或者确定无法找到为止。假设一次搜索，搜索了超过一个`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">，那么标记第一个`<font style="color:rgb(25, 27, 31);">SSTable`<font style="color:rgb(25, 27, 31);">搜索了一次，假设这种情况出现了多次，说明这个文件和多个其它的文件键范围有重叠
>

Seek Compaction 的流程如下：

1. 计算`SSTable`的最大`allowed_seeks`。

每次生成`SSTable`都要通过`Apply`方法应用到新的`Version`上面，所以 LevelDB 在`Apply`方法中计算`SSTable`允许 seek 的最大次数。

```cpp
f->allowed_seeks = static_cast<int>((f->file_size / 16384U));
if (f->allowed_seeks < 100) f->allowed_seeks = 100;
levels_[level].deleted_files.erase(f->number);
levels_[level].added_files->insert(f);
```

> 为什么设定没 16KB 数据一次 Compaction？
>
> LevelDB 假设：一次 seek 耗费 10ms，读或写 1MB 耗费 10ms，Compaction 1MB 数据需要 25MB 的 IO，那么 25 次 seek 的开销约等于一次 Compaction，一次 seek 约等于 Compaction 40KB 数据。所以 LevelDB 保守的选择每 16KB 数据触发一次 Compaction。
>

2. 消耗`allowed_seeks`

在 `DBImpl::Get` 方法中，当完成查询后会调用 `current->UpdateStats(stats)` 传递统计信息。<font style="color:rgb(34, 34, 34);">当查找文件而没有查找到时，`<font style="color:rgb(34, 34, 34);">allowed_seeks--`<font style="color:rgb(34, 34, 34);">，降为 0 时该文件标记到 `<font style="color:rgb(34, 34, 34);">file_to_compact_`<font style="color:rgb(34, 34, 34);">：

```cpp
bool Version::UpdateStats(const GetStats& stats) {
  FileMetaData* f = stats.seek_file;
  if (f != nullptr) {
    f->allowed_seeks--;
    if (f->allowed_seeks <= 0 && file_to_compact_ == nullptr) {
      file_to_compact_ = f;
      file_to_compact_level_ = stats.seek_file_level;
      return true;
    }
  }
  return false;
}
```

3. 执行 Seek Compaction

在 Compaction 的选择阶段也就是`VersionSet::PickCompaction`中，LevelDB 会根据`file_to_compact_`是否为空指针决定是否进行 Seek Compaction。（根据代码我们可以发现 Size Compaction 的优先级大于 Seek Compaction）

```cpp
else if (seek_compaction) {
    level = current_->file_to_compact_level_;
    c = new Compaction(options_, level);
    c->inputs_[0].push_back(current_->file_to_compact_);
}
```

#### 一些零散的 Compaction
1. 使用 Iterator

每当调用一次`it->Next()`或者`it->Prev()`移动迭代器时，迭代器内部都会调用一次`DBIter::ParseKey`，将当前Key解析出来。而在`DBIter::ParseKey`中，会定期采样当前 Key，看看这个Key 是否存在于多个`SSTable`中。如果是的话，就会将这个 Key 所在的 `SSTable`  的 `allowed_seeks--`，然后调用`MaybeScheduleCompaction`尝试触发 Compaction。这样做的目的是定期检查 `SSTable` 中的 Key 是否存在于多个 `SSTable` 中，如果是的话，就通过 Compaction 将这个 Key 所在的 `SSTable` 合并到更高 Level 的 `SSTable` 中，这样就可以减少 `SSTable` 的数量，提高读取效率。

2. 刚刚打开数据库

在`DBImpl::Open`中，会调用`MaybeScheduleCompaction`尝试触发 Compaction。刚打开数据的时候为什么需要尝试触发 Compaction 呢？因为当数据库上次关闭时，可能还有些没完成的 Compaction，比如 Compaction 进行中途机器断电了。所以当数据库打开时，需要尝试触发一次Compaction，检查下有没有未完成的 Compaction。

### 执行流程
1. 在可能需要 Compaction 的位置，LevelDB 会调用`MaybeScheduleCompaction`。
2. `MaybeScheduleCompaction`会进行一系列前置检查，并且通过`versions_->NeedsCompaction`判断是否需要 Compaction。

```cpp
// Returns true iff some level needs a compaction.
bool NeedsCompaction() const {
    Version* v = current_;
    return (v->compaction_score_ >= 1) || (v->file_to_compact_ != nullptr);
}
```

3. 若需要进行 Compaction，LevelDB 会通过后台线程执行`BackgroundCompaction`方法。`BackgroundCompaction`方法会计算出本次 Compaction 的范围，然后调用`DoCompactionWork`进行 Compaction。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739450962352-17da7620-95dc-49b4-8624-6e564bf2e98e.png)

接下来会研究`CompactRange`、`PickCompaction`和`DoCompaction`三个函数。

### 研究源码
#### CompactRange
手动触发的 Compaction，初始范围由用户指定，最终通过`versions_->CompactRange()`计算出 Compaction 的范围。

```cpp
Compaction* VersionSet::CompactRange(int level, const InternalKey* begin,
                                     const InternalKey* end) {
  std::vector<FileMetaData*> inputs;
  current_->GetOverlappingInputs(level, begin, end, &inputs);
  if (inputs.empty()) {
    return nullptr;
  }
  if (level > 0) {
    const uint64_t limit = MaxFileSizeForLevel(options_, level);
    uint64_t total = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
      uint64_t s = inputs[i]->file_size;
      total += s;
      if (total >= limit) {
        inputs.resize(i + 1);
        break;
      }
    }
  }

  Compaction* c = new Compaction(options_, level);
  c->input_version_ = current_;
  c->input_version_->Ref();
  c->inputs_[0] = inputs;
  SetupOtherInputs(c);
  return c;
}
```

+ 首先用`GetOverlappingInputs`得到`begin`到`end`范围内所有`SSTable`。
+ 对于 Level-0 以上层级， 限制每次 Compaction 的总文件大小，如果超出 Level-i 层的阈值，那么裁剪`SSTable`文件。
+ 最后将得到的裁剪范围和其他数据封装在`Compaction`对象中，`inputs_[0]`存放当前 Level 需要合并的`SSTable`
+ 最后用`SetupOtherInputs`方法根据 Level-i 得到 Level-i+1 的`SSTable`文件，并且存放在`inputs_[1]`中。

#### GetOverlappingInputs
讲了这么久的 LevelDB，我们还没仔细说说 LevelDB 是如何进行 Compact 操作的。所以在研究`GetOverlappingInputs`的源码之前，我们先用两个例子理解一下 LevelDB 如何进行 Compaction 的。

+ Level-0 的`SSTable`文件是无序可以重叠的，我们假设存在下列文件：

```plain
Level-0:
  SSTable1 [a-m] (seq 100)
  SSTable2 [f-t] (seq 95)  ← 更旧的数据
  SSTable3 [c-p] (seq 105) ← 最新的文件
  SSTable4 [k-z] (seq 102)

Level-1:
  SSTable5 [s-z] (已存在的老数据)
```

此时 Level-0 的`SSTable`数量达到阈值，触发了 Compaction。此时 Compaction 的初始范围是最新文件 SSTable3 的 `[f, t]`，经过多轮扫描范围扩大到了 `[a, z]`，合并文件扩大到了`inputs_ = [SSTable1, SSTable2 ,SSTable3 ,SSTable4 ,SSTable5]`。执行 Compact 之后会合并 Level-0 的全部文件生成新文件 SSTable6。

```plain
Level-0: (清空)
Level-1:
  SSTable5 [a-z] (旧文件)
  SSTable6 [s-z] (新文件)
```

+ Level-0 上层的`SSTable`文件是有序不重叠的：

既然上层 Level 的`SSTable`文件的 Key 都是不重叠的，那么是如何找到重叠的`SSTable`的呢？

我们沿用上一个例子，当 Level-0 完成合并生成了文件 SSTable6，此时 SSTable6 与之前 Level-1 的 SSTable5 就存在重叠部分（尽管 Level-1 是有序不重叠的），此时我们就需要对 Level-1 扩大合并范围。

```plain
Level-2:
  SSTable1 [a-f] (seq 100)
  SSTable2 [i-t] (seq 95)
  SSTable3 [u-z] (seq 102)
  SSTable4 [b-p] (seq 105) ← levle-1 刚刚生成的新文件,存在重复部分

Level-3:
  SSTable5 [p-y] (已存在的老数据)
```



现在我们再来看看`GetOverlappingInputs`的代码：

```cpp
// Store in "*inputs" all files in "level" that overlap [begin,end]
void Version::GetOverlappingInputs(int level, const InternalKey* begin,
const InternalKey* end,
std::vector<FileMetaData*>* inputs) {
    assert(level >= 0);
    assert(level < config::kNumLevels);
    inputs->clear();
    Slice user_begin, user_end;
    if (begin != nullptr) {
        user_begin = begin->user_key();
    }
    if (end != nullptr) {
        user_end = end->user_key();
    }
    const Comparator* user_cmp = vset_->icmp_.user_comparator();
    for (size_t i = 0; i < files_[level].size();) {
        FileMetaData* f = files_[level][i++];
        const Slice file_start = f->smallest.user_key();
        const Slice file_limit = f->largest.user_key();
        if (begin != nullptr && user_cmp->Compare(file_limit, user_begin) < 0) {
            // "f" is completely before specified range; skip it
        } else if (end != nullptr && user_cmp->Compare(file_start, user_end) > 0) {
            // "f" is completely after specified range; skip it
        } else {
            inputs->push_back(f);
            if (level == 0) {
                // Level-0 files may overlap each other.  So check if the newly
                // added file has expanded the range.  If so, restart search.
                if (begin != nullptr && user_cmp->Compare(file_start, user_begin) < 0) {
                    user_begin = file_start;
                    inputs->clear();
                    i = 0;
                } else if (end != nullptr &&
                    user_cmp->Compare(file_limit, user_end) > 0) {
                    user_end = file_limit;
                    inputs->clear();
                    i = 0;
                }
            }
        }
    }
}
```

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739528060788-1ab35d51-029e-41ce-b471-2504505c8192.png)

由于 Level-0 的`SSTable`文件是无序且可能存在重叠的，所以当发现某个文件的 Key 范围比当前范围更宽时（比如文件起始 < 当前起始），说明之前的扫描可能漏掉了其他文件。

> 我在研究源码时候还有一个问题，既然 Level-0 会进行动态范围扩展，那直接将`user_begin`设置为所有`SSTable`中最小的`file_start`不就好了吗？或者说 Level-0 的 Compaction 是否会合并全部的`SSTable`文件？
>
> Level-0 的 Compaction 不一定会合并全部文件，反例是当各个`SSTable`的 Key 离散的时候。
>

#### SetupOtherInputs
`SetupOtherInputs`方法的目的就是根据 Level-i 的范围，计算出 Level-i+1 需要加入哪些文件。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739607547099-4b466574-d7e8-4375-b346-d302607affc2.png)

<font style="color:rgb(0, 0, 0);">选择level-i+1 的步骤如下：

1. <font style="color:rgb(0, 0, 0);">根据 Level-i 加入的文件，计算得到最小 Key 和最大 Key。
2. <font style="color:rgb(0, 0, 0);">根据最小 Key 和最大 Key 的范围，从 Level-i+1 中选择和该范围有重叠的所有文件，计入`<font style="color:rgb(0, 0, 0);">inputs_[1]`<font style="color:rgb(0, 0, 0);">中。
3. <font style="color:rgb(0, 0, 0);">根据第一步和第二步得到的所有的 Level-i 的文件和 Level-i+1 的文件，重新计算新的最小 Key 和新的最大 Key。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739608156171-59e0fc90-15d8-4047-a027-2d05a05cd5e5.png)

那接下来这一大坨代码是在干什么呢？我们借助下图解释：

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1739607931357-7ebf9e14-c477-4d25-bbdb-dfe740df1a5a.png)

假如当前 Level-i 已经选择了文件 H，那么 Level-i+1 层应该会扩充到文件 B、文件 C 和文件 D，这时候 Key 的范围扩大了，那么 Level-i 层是否需要再次扩充文件 G？上图的情况应该是不需要的，因为如果 Level-i 扩充了文件 G，那么 Key 再次扩大，Level-i+1 的文件 A 也理应加入，这样就可能陷入循环。所以上面代码就是为了判断我们能否在 Level 中增加文件数量而不增加 Level-i+1 文件的数量。

#### PickCompaction
`PickCompaction`方法是用于计算出参与此次 Compaction 的 Level-i 层的`SSTable`文件，根据 Seek Compaction 或者 Size Compaction 有不同的计算方法。

Seek Compaction 的计算方式很简单，哪个文件的 Seek Miss 次数超过阈值，那么就将哪个文件加入`inputs_`。Size Compaction 需要考虑当前 Level 上一次 Compaction 到哪一个 Key，将大于该 Key 的第一个文件作为这次 Compaction 的对象。最后通过`SetupOtherInputs`扩充上层 Level 的范围。

> PS：由于 Level-0 的`SSTable`无序且可能重叠，所以 Level-0 会通过`GetOverlappingInputs`扩充范围，不一定只有一个文件。
>

```cpp
Compaction* VersionSet::PickCompaction() {
  Compaction* c;
  int level;

  // We prefer compactions triggered by too much data in a level over
  // the compactions triggered by seeks.
  const bool size_compaction = (current_->compaction_score_ >= 1);
  const bool seek_compaction = (current_->file_to_compact_ != nullptr);
  if (size_compaction) {
    level = current_->compaction_level_;
    assert(level >= 0);
    assert(level + 1 < config::kNumLevels);
    c = new Compaction(options_, level);

    // Pick the first file that comes after compact_pointer_[level]
    for (size_t i = 0; i < current_->files_[level].size(); i++) {
      FileMetaData* f = current_->files_[level][i];
      if (compact_pointer_[level].empty() ||
          icmp_.Compare(f->largest.Encode(), compact_pointer_[level]) > 0) {
        c->inputs_[0].push_back(f);
        break;
      }
    }
    if (c->inputs_[0].empty()) {
      // Wrap-around to the beginning of the key space
      c->inputs_[0].push_back(current_->files_[level][0]);
    }
  } else if (seek_compaction) {
    level = current_->file_to_compact_level_;
    c = new Compaction(options_, level);
    c->inputs_[0].push_back(current_->file_to_compact_);
  } else {
    return nullptr;
  }

  c->input_version_ = current_;
  c->input_version_->Ref();

  // Files in level 0 may overlap each other, so pick up all overlapping ones
  if (level == 0) {
    InternalKey smallest, largest;
    GetRange(c->inputs_[0], &smallest, &largest);
    // Note that the next call will discard the file we placed in
    // c->inputs_[0] earlier and replace it with an overlapping set
    // which will include the picked file.
    current_->GetOverlappingInputs(0, &smallest, &largest, &c->inputs_[0]);
    assert(!c->inputs_[0].empty());
  }

  SetupOtherInputs(c);

  return c;
}

```

#### DoCompaction
前面我们了解了 Manual Compaction 和 Auto Compaction 下计算 Compaction 文件的方法，现在就需要研究具体是如何 Compaction 的了。

```cpp
Status DBImpl::DoCompactionWork(CompactionState* compact) {
  // 记录压缩开始时间
  const uint64_t start_micros = env_->NowMicros();
  int64_t imm_micros = 0;  // 用于统计不可变memtable的压缩时间

  // 记录日志：显示要压缩的L层和L+1层文件数量
  Log(options_.info_log, "Compacting %d@%d + %d@%d files",
      compact->compaction->num_input_files(0), compact->compaction->level(),
      compact->compaction->num_input_files(1),
      compact->compaction->level() + 1);

  // 断言确保当前层有文件需要压缩
  assert(versions_->NumLevelFiles(compact->compaction->level()) > 0);
  assert(compact->builder == nullptr);  // 确保builder未初始化
  assert(compact->outfile == nullptr);   // 确保输出文件未打开

  // 确定最小有效快照版本号（用于数据过期判断）
  if (snapshots_.empty()) {
    compact->smallest_snapshot = versions_->LastSequence();
  } else {
    compact->smallest_snapshot = snapshots_.oldest()->sequence_number();
  }

  // 创建合并迭代器（合并所有待压缩文件的键值）
  Iterator* input = versions_->MakeInputIterator(compact->compaction);

```

这段代码首先通过断言保证了程序的稳定性，然后根据待 Compact 的文件生成一个迭代器，注意这个迭代器是**有序的且**`**SequenceNumber**`**递减的**。（比较器的代码在可以注意到`SequenceNumber`大的更小）

```cpp
int InternalKeyComparator::Compare(const Slice& akey, const Slice& bkey) const {
  // Order by:
  //    increasing user key (according to user-supplied comparator)
  //    decreasing sequence number
  //    decreasing type (though sequence# should be enough to disambiguate)
  int r = user_comparator_->Compare(ExtractUserKey(akey), ExtractUserKey(bkey));
  if (r == 0) {
    const uint64_t anum = DecodeFixed64(akey.data() + akey.size() - 8);
    const uint64_t bnum = DecodeFixed64(bkey.data() + bkey.size() - 8);
    if (anum > bnum) {
      r = -1;
    } else if (anum < bnum) {
      r = +1;
    }
  }
  return r;
}
```

下面这部分代码是 `DoCompactionWork`方法的核心逻辑：

```cpp
// ... 其他代码 ...

while (input->Valid() && !shutting_down_.load(std::memory_order_acquire)) {
    // ... 优先处理immutable memtable的代码 ...

    // 解析内部键（包含用户键+序列号+类型）
    Slice key = input->key();
    if (!ParseInternalKey(key, &ikey)) {
        // 解析失败视为错误键，直接清除状态
        current_user_key.clear();
        has_current_user_key = false;
        last_sequence_for_key = kMaxSequenceNumber;
    } else {
        // 判断是否是新的用户键
        if (!has_current_user_key ||
            user_comparator()->Compare(ikey.user_key, Slice(current_user_key)) != 0) {
            // 新的用户键：重置状态
            current_user_key.assign(ikey.user_key.data(), ikey.user_key.size());
            has_current_user_key = true;
            last_sequence_for_key = kMaxSequenceNumber; // 初始设为最大值
        }

        // 判断是否应该丢弃该键
        if (last_sequence_for_key <= compact->smallest_snapshot) {
            // 规则A：存在更新的版本，且该版本对现存快照可见
            drop = true;
        } else if (ikey.type == kTypeDeletion &&
                ikey.sequence <= compact->smallest_snapshot &&
                compact->compaction->IsBaseLevelForKey(ikey.user_key)) {
            // 规则B：可安全删除的墓碑标记
            drop = true;
        }

        // 更新当前键的最新序列号
        last_sequence_for_key = ikey.sequence;
    }
}
```

由于 Compaction 消耗时间长，所以每次写入一个 Key 都会判断 LevelDB 是否关闭、是否需要进行 Minor Compaction（Minor Compaction 优先级大于 Major Compaction）。之后会判断 Key 是否需要 drop。判断的规则如下：如果某个 `user_key` 的非最新版本小于快照版本，则可以直接丢弃，因为读最新的版本就足够了；如果某个删除操作的版本小于快照版本，并且在更高层没有相同的 `user_key`，那么这个删除操作及其之前更早的插入操作可以同时丢弃了。

```cpp
if (!drop) {
      // Open output file if necessary
      if (compact->builder == nullptr) {
        status = OpenCompactionOutputFile(compact);
        if (!status.ok()) {
          break;
        }
      }
      if (compact->builder->NumEntries() == 0) {
        compact->current_output()->smallest.DecodeFrom(key);
      }
      compact->current_output()->largest.DecodeFrom(key);
      compact->builder->Add(key, input->value());

      // Close output file if it is big enough
      if (compact->builder->FileSize() >=
          compact->compaction->MaxOutputFileSize()) {
        status = FinishCompactionOutputFile(compact, input);
        if (!status.ok()) {
          break;
        }
      }
}
input->Next();
```

LevelDB 直到遇到第一个需要保留的 Key 时才创建输出文件，将 Key 插入`SSTable`，并且在生成的`SSTable`文件更新最大最小 Key 用于查询与后续的层级管理。如果`SSTable`文件大小达到`MaxOutputFileSize`也就是 2MB，LevelDB 会调用`FinishCompactionOutputFile`完成当前`SSTable`文件的创建。

> `FinishCompactionOutputFile`方法实际上就是将`SSTable`文件落盘，关闭文件句柄，并且将创建的文件记录到当前版本的`VersionEdit`上。
>

至此，Compaction 部分全部结束。



## 版本控制
### 基本概念
上一节 Compaction 部分学习中，不知道你有没有注意到我们没有讲解 Compaction 后废弃的`SSTable`是如何处理的。我们在进行查询操作时，还能进行 Compaction 操作吗？如果`SSTable`正在进行合并，是否会影响查询？我们通过上面的问题，来了解 LevelDB 的版本控制。

我们看看 LevelDB 查询函数的代码：

```cpp
Status DBImpl::Get(const ReadOptions& options, const Slice& key,
                   std::string* value) {

  // ... 其它代码 ...
  Version* current = versions_->current();
  mem->Ref();
  if (imm != nullptr) imm->Ref();
  current->Ref();

  bool have_stat_update = false;
  Version::GetStats stats;

  // Unlock while reading from files and memtables
  {
    mutex_.Unlock();
    // First look in the memtable, then in the immutable memtable (if any).
    LookupKey lkey(key, snapshot);
    if (mem->Get(lkey, value, &s)) {
      // Done
    } else if (imm != nullptr && imm->Get(lkey, value, &s)) {
      // Done
    } else {
      s = current->Get(options, lkey, value, &stats);
      have_stat_update = true;
    }
    mutex_.Lock();
  }
```

可以注意到 LevelDB 通过`mem_Get()`、`imm->Get()`和 `current->Get()`来查询，前两个分别是 MemTable 和 Immutabe MemTable，最后一个是什么呢？`current->Get()`查询的是`SSTable`。

`current`是 Version 的实例，也就是当前版本。

在解释版本控制是什么玩意之前，我们先回答几个问题。我们说过 LevelDB 将 `Immutable Writable File` 持久化到磁盘中会转变为`SSTable`，并且`SSTable`有不同的 level，那 LevelDB 是如何区分哪个`SST`文件是哪个 level 呢？LevelDB 是如何知道哪个关键字在哪一个文件呢？版本就是 LevelDB 数据库的元信息，LevelDB 需要这些信息才能<font style="color:rgb(25, 27, 31);">快速的从SSTable里读取出相应的键的值。

<font style="color:rgb(25, 27, 31);">关于版本控制，我们主要关注`<font style="color:rgb(25, 27, 31);">Version`<font style="color:rgb(25, 27, 31);">，`<font style="color:rgb(25, 27, 31);">VersionEdit`<font style="color:rgb(25, 27, 31);">，`<font style="color:rgb(25, 27, 31);">VersionSet`<font style="color:rgb(25, 27, 31);">三个类。

先来说说Version，Version表示了当前 LevelDB 的版本信息，版本信息内容包括：当前每一层的`SSTable`文件元信息、Seek Miss 次数过多的文件的元信息，以及文件所在的 Level、记录所有 Level 中`compaction_score_ `最大的那一层及其level，这些参数都是服务于 Compaction 的。

```cpp
class Version{
private:
      VersionSet* vset_;
      Version* next_;
      Version* prev_; 
      int refs_;
      std::vector<FileMetaData*> files_[config::kNumLevels];
      FileMetaData* file_to_compact_;
      int file_to_compact_level_;
      double compaction_score_;
      int compaction_level_;    
};
```

`SSTable`文件元信息结构由数据结构 `FileMetaData` 来表示，保存的数据有：当前文件被引用了多少次、文件允许被Seek的次数、文件编号、文件大小、文件最大 Key 和最小 Key。

```cpp
struct FileMetaData {
  FileMetaData() : refs(0), allowed_seeks(1 << 30), file_size(0) {}
  int refs;
  int allowed_seeks;  // Seeks allowed until compaction
  uint64_t number;
  uint64_t file_size;    // File size in bytes
  InternalKey smallest;  // Smallest internal key served by table
  InternalKey largest;   // Largest internal key served by table
};
```

+ VersionEdit

每次`SSTable`文件发生变化时（例如一次），都会更新当前版本。为了缩小Version切换的时间点，LevelDB 将这些操作都封装到 VersionEdit 中，最后通过 builder 将 VersionEdit 应用到当前 Version，生成新的 Version。

```cpp
class VersionEdit{
private:
  std::string comparator_;
  uint64_t log_number_;
  uint64_t prev_log_number_;
  uint64_t next_file_number_;
  SequenceNumber last_sequence_;
  bool has_comparator_;
  bool has_log_number_;
  bool has_prev_log_number_;
  bool has_next_file_number_;
  bool has_last_sequence_;
  std::vector<std::pair<int, InternalKey>> compact_pointers_;
  DeletedFileSet deleted_files_;
  std::vector<std::pair<int, FileMetaData>> new_files_;
};
```

+ VersionSet

由于 LevelDB 中采用了 LSM-Tree 这个数据结构，所以在 LevelDB 中同时可能存在多个 Version。VersionSet 是一个 Version 构成的双向链表，这些 Version 按时间顺序先后产生，记录了当时的元信息，链表头指向当前最新的 Version，同时维护了每个 Version 的引用计数，被引用中的 Version 不会被删除，其对应的 `SSTable` 文件也因此得以保留，通过这种方式，使得 LevelDB 可以在一个稳定的快照视图上访问文件。VersionSet 中除了 Version 的双向链表外还会记录一些如 `LogNumber`、`Sequence`，下一个 `SSTable` 文件编号的状态信息。

> LSM-Tree 是一种专门优化写入性能的数据存储结构， 在内存上通过跳表进行存储，在磁盘上用多层`SSTable`进行存储（实际上就是 LevelDB 的结构）。
>

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1737903777798-2e3ef76f-1199-4ca2-8e01-8128b81e782d.png?x-oss-process=image%2Fformat%2Cwebp)

了解了 Version、VersionEdit 和 VersionSet，你应该就大致明白前面几个问题的答案了。当 LevelDB 正在进行 Compaction 时，对`SSTable`文件进行合并，生成 VersionEdit 并且应用到当前版本生成新的 Version。与此同时如果我们在进行查询操作，我们会对原始的 Version 进行查询，会根据它保存的`SSTable`元信息进行查询。并且由于原始 Version 还存在引用，它记录的`SSTable`文件不会被删除。

上面 Version 的析构代码也可以印证这一点：

```cpp
Version::~Version() {
  assert(refs_ == 0);

  // Remove from linked list
  prev_->next_ = next_;
  next_->prev_ = prev_;

  // Drop references to files
  for (int level = 0; level < config::kNumLevels; level++) {
    for (size_t i = 0; i < files_[level].size(); i++) {
      FileMetaData* f = files_[level][i];
      assert(f->refs > 0);
      f->refs--;
      if (f->refs <= 0) {
        delete f;
      }
    }
  }
}
```

当 Version 的引用计数`ref_==0`时，Version 将自己从 VersionSet 维护的环形双向链表中移除，然后遍历自己所持有的`SSTable`文件的元信息使其引用自减，如果某一个 FileMetaData 当前唯一的引用对象就是该 Version, 那么将这个 FileMetaData 删除掉，避免内存泄漏。

### <font style="color:rgb(0, 0, 0);">VersionSet::Builder
在进行 Minor Compaction 或者 Marjor Compaction 的时候产生一个 VersionEdit，Builder 会将它应用到当前 Version 上生成新的 Version。接下来我们分析一下这个过程：

+ Minor Compaction

```cpp
Status DBImpl::WriteLevel0Table(MemTable* mem, VersionEdit* edit,
                                Version* base) {
    // 将 Immutable Memtable 写入到 SSTable
    if (s.ok() && meta.file_size > 0) {
        edit->AddFile(level, meta.number, meta.file_size, meta.smallest, meta.largest);
    }
}
```

+ Major Compaction

```cpp
Status DBImpl::InstallCompactionResults(CompactionState* compact) {
  
    // 将删除的文件添加入 VersionEdit
    compact->compaction->AddInputDeletions(compact->compaction->edit());
    const int level = compact->compaction->level();

    // 将新增的文件添加入 VersionEdit
    for (size_t i = 0; i < compact->outputs.size(); i++) {
      const CompactionState::Output& out = compact->outputs[i];
      compact->compaction->edit()->AddFile(level + 1, out.number, out.file_size,
                                         out.smallest, out.largest);
    }
    return versions_->LogAndApply(compact->compaction->edit(), &mutex_);
}
```

新增文件和删除文件都是调用了`edit->addFile()`和`edit->RemoveFile()`方法，本质是也就是将`SSTable`的元信息`FileMetaData`插入`new_files_`和`deleted_files_`数组中。

```cpp
// AddFile 需要保存完整的文件元数据信息(FileMetaData)
// RemoveFile 只需要知道删除哪个文件就好了
void AddFile(int level, uint64_t file, uint64_t file_size,
               const InternalKey& smallest, const InternalKey& largest) {
    FileMetaData f;
    f.number = file;
    f.file_size = file_size;
    f.smallest = smallest;
    f.largest = largest;
    new_files_.push_back(std::make_pair(level, f));
}


 void RemoveFile(int level, uint64_t file) {
    deleted_files_.insert(std::make_pair(level, file));
}
```

新增或删除文件被记录文件之后，Builder 会在`VersionSet::LogAndApply`和`VersionSet::Recvoer`时被调用。`VersionSet::Recover`会在 Manifest 部分讲解，所以我们讲讲`VersionSet::LogAndApply`。

当 Minor Compaction 完成或者 Major Compaction 完成时，`VersionSet::LogAndApply`就会被调用，将 VersionEdit 持久化并应用到当前版本，生成新版本。

1. 将`edit`应用在`current_`，生成一个新的 Version。

```cpp
Version* v = new Version(this);
{
    // 基于当前版本创建 builder
    Builder builder(this, current_);
    // 应用新的修改
    builder.Apply(edit);
    // 保存为新的 Version
    builder.SaveTo(v);
}
```

```cpp
void Apply(const VersionEdit* edit) {
    // 计算压缩点
    for (size_t i = 0; i < edit->compact_pointers_.size(); i++) {
      const int level = edit->compact_pointers_[i].first;
      vset_->compact_pointer_[level] =
          edit->compact_pointers_[i].second.Encode().ToString();
    }

    // 记录删除文件
    for (const auto& deleted_file_set_kvp : edit->deleted_files_) {
      const int level = deleted_file_set_kvp.first;
      const uint64_t number = deleted_file_set_kvp.second;
      levels_[level].deleted_files.insert(number);
    }

    // 记录新增文件，同时计算文件允许seek的最大次数
    for (size_t i = 0; i < edit->new_files_.size(); i++) {
      const int level = edit->new_files_[i].first;
      FileMetaData* f = new FileMetaData(edit->new_files_[i].second);
      f->refs = 1;

      f->allowed_seeks = static_cast<int>((f->file_size / 16384U));
      if (f->allowed_seeks < 100) f->allowed_seeks = 100;

      levels_[level].deleted_files.erase(f->number);
      levels_[level].added_files->insert(f);
    }
  }

```

压缩点`compact_pointer_`是什么呢？我们回忆一下`PickCompaction`的代码，对于数据大小触发的 Size Compaction，LevelDB 会选取 `compact_pointer_` 后的第一个文件作为 Compaction 对象，即本层上一次 Compaction 区间之后的文件，有助于避免重启后从头压缩。

```cpp
Compaction* VersionSet::PickCompaction() {
    if (size_compaction) {    
        // Pick the first file that comes after compact_pointer_[level]
        for (size_t i = 0; i < current_->files_[level].size(); i++) {
          FileMetaData* f = current_->files_[level][i];
          if (compact_pointer_[level].empty() ||
              icmp_.Compare(f->largest.Encode(), compact_pointer_[level]) > 0) {
            c->inputs_[0].push_back(f);
            break;
          }
        }
    }
}
```

2. 计算新 Version 下，需要进行Major Compaction 的文件

```cpp
Compaction* VersionSet::PickCompaction(){
    // builder
    Finalize(v);
    // manifest and version edit
}
```

在 Size Compaction 的触发时机那一部分我们已经介绍过了`Finalize`方法，它用于计算每一层的`score`，Size Compaction 将发生在分数最大的那一层。

3. 更新元信息

```cpp
  std::string new_manifest_file;
  Status s;
  if (descriptor_log_ == nullptr) {
    // No reason to unlock *mu here since we only hit this path in the
    // first call to LogAndApply (when opening the database).
    assert(descriptor_file_ == nullptr);
    new_manifest_file = DescriptorFileName(dbname_, manifest_file_number_);
    s = env_->NewWritableFile(new_manifest_file, &descriptor_file_);
    if (s.ok()) {
      descriptor_log_ = new log::Writer(descriptor_file_);
      s = WriteSnapshot(descriptor_log_);
    }
  }

  // Unlock during expensive MANIFEST log write
  {
    mu->Unlock();

    // Write new record to MANIFEST log
    if (s.ok()) {
      std::string record;
      edit->EncodeTo(&record);
      s = descriptor_log_->AddRecord(record);
      if (s.ok()) {
        s = descriptor_file_->Sync();
      }
      if (!s.ok()) {
        Log(options_->info_log, "MANIFEST write: %s\n", s.ToString().c_str());
      }
    }

    // If we just created a new descriptor file, install it by writing a
    // new CURRENT file that points to it.
    if (s.ok() && !new_manifest_file.empty()) {
      s = SetCurrentFile(env_, dbname_, manifest_file_number_);
    }
```

这一部分代码会创建 Manifest 文件（如果不存在的话），并且将 VersionEdit 序列化记录到 Manifest 文件中（关于 Manifest 的部分后面会开一个新专题细讲）。

4. 调用`AppendVersion`， 将新 Version 插入双向链表中

```cpp
if (s.ok()) {
    AppendVersion(v);
    log_number_ = edit->log_number_;
    prev_log_number_ = edit->prev_log_number_;
} else {
    delete v;
    if (!new_manifest_file.empty()) {
        delete descriptor_log_;
        delete descriptor_file_;
        descriptor_log_ = nullptr;
        descriptor_file_ = nullptr;
        env_->RemoveFile(new_manifest_file);
    }
}
```

`VersionSet::AppendVersion`做的事情其实就是更新`current_`和双向链表前驱后继。

```cpp
void VersionSet::AppendVersion(Version* v) {
  // Make "v" current
  assert(v->refs_ == 0);
  assert(v != current_);
  if (current_ != nullptr) {
    current_->Unref();
  }
  current_ = v;
  v->Ref();

  // Append to linked list
  v->prev_ = dummy_versions_.prev_;
  v->next_ = &dummy_versions_;
  v->prev_->next_ = v;
  v->next_->prev_ = v;
}
```

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1740205194778-d9ec6886-ea7d-497f-8b17-c0e39d165004.png)

至此就完成了`LogAndApply`的部分，了解了 VersionEdit 是如何应用的，版本控制专题也暂时告一段落。

## 崩溃恢复
### Manifest
Manifest 文件本质上也是日志文件，格式和 LOG 是相同的。

+ 第一条记录是全量的 LevelDB 版本信息
+ 后续每一条记录都是 LevelDB 版本变更信息

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1740211464138-ab0250c7-69c2-405d-bfde-0cfb9949e231.png)

#### 创建 Manifest 文件
在`DBImpl::NewDB()`和`VersionSet::LogAndApply()`方法中，LevelDB 都有可能创建新的 Manifest 文件。

```cpp
std::string new_manifest_file;
Status s;
if (descriptor_log_ == nullptr) {
    // No reason to unlock *mu here since we only hit this path in the
    // first call to LogAndApply (when opening the database).
    assert(descriptor_file_ == nullptr);
    new_manifest_file = DescriptorFileName(dbname_, manifest_file_number_);
    s = env_->NewWritableFile(new_manifest_file, &descriptor_file_);
}

std::string DescriptorFileName(const std::string& dbname, uint64_t number) {
  assert(number > 0);
  char buf[100];
  std::snprintf(buf, sizeof(buf), "/MANIFEST-%06llu",
                static_cast<unsigned long long>(number));
  return dbname + buf;
}
```

`DescriptorFileName`方法负责为 Manifest 文件命名，之后通过底层的`WriteableFile`创建 Manifest 文件。

但有所不同的是，`VersionSet::LogAndApply()`创建新 Manifest 时会通过 WriteSnapshot 写入完整的当前状态，因为此时数据库已经有了完整的文件结构；而 `DBImpl::NewDB()` 时数据库是空的,不需要写入完整状态，写入一个初始的 VersionEdit。

#### 写入版本变更
每次 Compaction 之后都会导致`SSTable`文件的变动，触发版本变更。`VersionSet::LogAndApply()`会将 VersionEdit 序列化为 `record` 写入 Manifest 文件。

```cpp
// Unlock during expensive MANIFEST log write
{
    mu->Unlock();
    // Write new record to MANIFEST log
    if (s.ok()) {
      std::string record;
      edit->EncodeTo(&record);
      s = descriptor_log_->AddRecord(record);
      if (s.ok()) {
        s = descriptor_file_->Sync();
      }
      if (!s.ok()) {
        Log(options_->info_log, "MANIFEST write: %s\n", s.ToString().c_str());
      }
}
```

> 注意，对于 LevelDB 来说，增减某些`SSTable`文件需要作为一个原子性操作，状态变更前后需要保持数据库的一致性。在整个过程中，原子性体现在：整个操作的完成标志为 Manifest 文件中完整的写入了一条 Record，在此之前，即便某些文件写入失败导致进程退出，数据库重启启动时，仍然能够恢复到崩溃之前正确的状态，而将这些无用的`SSTable`文件删除，重新进行 Compaction。一致性体现在：LevelDB 状态变更的操作都是以 Version 更新为标记，而 Version 更新是整个流程的最后一步，因此数据库必然都是从一个一致性的状态变更到另外一个一致性的状态。
>

### <font style="color:rgb(64, 64, 64);background-color:rgb(231, 242, 250);">Current
当数据库文件损坏或者 Manifest 文件过大时，会创建新的 Manifest 文件，也就是说同一时间 LevelDB 可能存在多个 Manifest 文件，所以 LevelDB 通过 Current 文件来知道正在使用的 Manifest 文件名。

```cpp
// If we just created a new descriptor file, install it by writing a
// new CURRENT file that points to it.
if (s.ok() && !new_manifest_file.empty()) {
    s = SetCurrentFile(env_, dbname_, manifest_file_number_);
}
```

### 恢复版本信息
LevelDB 每次打开都会有一个 Recover 的过程，简要地来说就是利用 Manifest 信息重新构建一个最新的 Version。一方面它会恢复版本信息，另一方面它会回放 WAL 中记录的操作，恢复 `MemTable`里面的内容。

恢复的主逻辑位于`VersionSet::Recover`中，流程如下：

+ 读取 Current 文件的内容找到 Manifest 文件名

```cpp
 // Read "CURRENT" file, which contains a pointer to the current manifest file
  std::string current;
  Status s = ReadFileToString(env_, CurrentFileName(dbname_), &current);
  if (!s.ok()) {
    return s;
  }
  if (current.empty() || current[current.size() - 1] != '\n') {
    return Status::Corruption("CURRENT file does not end with newline");
  }
  current.resize(current.size() - 1);

  std::string dscname = dbname_ + "/" + current;
  SequentialFile* file;
  s = env_->NewSequentialFile(dscname, &file);
  if (!s.ok()) {
    if (s.IsNotFound()) {
      return Status::Corruption("CURRENT points to a non-existent file",
                                s.ToString());
    }
    return s;
  }
```

+ 遍历 Manifest 文件，将所有记录 Apply 到 Version Builder 上

```cpp
LogReporter reporter;
reporter.status = &s;
log::Reader reader(file, &reporter, true /*checksum*/,
                       0 /*initial_offset*/);
Slice record;
std::string scratch;
while (reader.ReadRecord(&record, &scratch) && s.ok()) {
    ++read_records;
    VersionEdit edit;
    s = edit.DecodeFrom(record);
    if (s.ok()) {
    if (edit.has_comparator_ &&
        edit.comparator_ != icmp_.user_comparator()->Name()) {
        s = Status::InvalidArgument(
            edit.comparator_ + " does not match existing comparator ",
         icmp_.user_comparator()->Name());
        }
    }
    if (s.ok()) {
        builder.Apply(&edit);
    }
    // ...
}
```

+ 从 Builder 中获得最终的 Version，将它加入 VersionSet，并作为当前版本使用

```cpp
Version* v = new Version(this);
builder.SaveTo(v);
// Install recovered version
 Finalize(v);
AppendVersion(v);
manifest_file_number_ = next_file;
next_file_number_ = next_file + 1;
last_sequence_ = last_sequence;
log_number_ = log_number;
prev_log_number_ = prev_log_number;

```

+ 检查现存的 Manifest 文件能否重用

如果旧的 Manifest 文件大小不是太大就可以重用它（重启是唯一一处 Manifest 文件大小缩小的地方，我们不想让 MANIFEST 文件大小无限增长）。如果不能重用，就会提交一条 Manifest 记录到新的 Manifest 文件里，包含了当前版本的全量信息。

```cpp
bool VersionSet::ReuseManifest(const std::string& dscname,
                               const std::string& dscbase) {
  if (!options_->reuse_logs) {
    return false;
  }
  FileType manifest_type;
  uint64_t manifest_number;
  uint64_t manifest_size;
  if (!ParseFileName(dscbase, &manifest_number, &manifest_type) ||
      manifest_type != kDescriptorFile ||
      !env_->GetFileSize(dscname, &manifest_size).ok() ||
      // Make new compacted MANIFEST if old one is too big
      manifest_size >= TargetFileSize(options_)) {
    return false;
  }

  assert(descriptor_file_ == nullptr);
  assert(descriptor_log_ == nullptr);
  Status r = env_->NewAppendableFile(dscname, &descriptor_file_);
  if (!r.ok()) {
    Log(options_->info_log, "Reuse MANIFEST: %s\n", r.ToString().c_str());
    assert(descriptor_file_ == nullptr);
    return false;
  }

  Log(options_->info_log, "Reusing MANIFEST %s\n", dscname.c_str());
  descriptor_log_ = new log::Writer(descriptor_file_, manifest_size);
  manifest_file_number_ = manifest_number;
  return true;
}
```

### MemTable 恢复
前面我们说到，LevelDB 采用 WAL 的方式记录数据。由于 Minor Compaction 在一定条件下才触发，出现故障时内存中的 MemTable 可能还没有持久化到磁盘，成为 SSTable-0。这时候我们就需要通过 LOG 恢复 MemTable 中的数据。

在 LevelDB 中，`logNumber`是日志文件的唯一标识符。每个日志文件（.log文件）都有一个独特的编号，这个编号就是`logNumber`。这些日志文件用于记录数据库的写入操作，确保在崩溃后能够恢复数据。

logNumber在以下几个关键时刻被记录：

+ 创建新数据库时：`DBImpl::NewDB` 中，设置初始LogNumber为0：

```cpp
new_db.SetLogNumber(0);
```

+ 打开数据库创建新日志文件时：在`DB::Open`函数中：

```cpp
edit.SetLogNumber(new_log_number);
impl->logfile_number_ = new_log_number;
```

+ 切换到新日志文件时：当 MemTable 满时，在创建新的日志文件时会分配并记录新的`logNumber`：

```cpp
uint64_t new_log_number = versions_->NewFileNumber();
logfile_number_ = new_log_number;
```

+ 写入 MANIFEST 文件时：在 VersionEdit 中记录`logNumber`，然后通过`VersionSet::LogAndApply`写入持久化存储：

```cpp
edit.SetLogNumber(logfile_number_); 
```



LevelDB 恢复 MemTable 的步骤如下：

+ 从 Manifest 中读出最近的 MinLogNumber
+ 从 Version 中读出现存的全部文件
+ 遍历所有文件找到 LOG 文件，并且 LogNumber 大于 MinLogNumber 的日志
+ 逐个恢复日志

```cpp
Status DBImpl::Recover(VersionEdit* edit, bool* save_manifest) {
    env_->CreateDir(dbname_);  // 创建数据库目录
    // 加文件锁，防止其他进程进入
    Status s = env_->LockFile(LockFileName(dbname_), &db_lock_);

    if (!env_->FileExists(CurrentFileName(dbname_))) {
        // 如果CURRENT文件不存在，说明需要新创建数据库
        s = NewDB();
    }

    // 读取MANIFEST文件进行版本信息的恢复
    s = versions_->Recover(save_manifest);

    // 之前的MANIFEST恢复，会得到版本信息，里面包含了之前的log number
    // 搜索文件系统里的log，如果这些日志的编号 >= 这个log number，那么这些
    // 日志都是关闭时丢失的数据，需要恢复，这里将日志按顺序存储在logs里面

    // 逐个恢复日志的内容
    for (size_t i = 0; i < logs.size(); i++) {
        s = RecoverLogFile(logs[i], (i == logs.size() - 1), save_manifest, edit, &max_sequence);
        // ...
    }
    // ...
    return Status::OK();
}
```

`RecoverLogFile()`方法就是通过`log::Reader`读出日志内容，再通过`WriteBatch`方法回放写操作，将数据写入 MemTable 中。需要注意，`RecoverLogFile`方法在恢复 MemTable 的过程中，如果 MemTable 超过大小，会进行 Minor Compaction。

> Q：恢复 MemTable 时为什么会超过大小？在写入时如果 MemTable 超过大小不就直接 compact 吗？
>
> A：研究源码我们可以看到`write_buffer_size`属性隶属于`options_`对象，他是 LevelDB 的一个配置项，也就是说每次打开 LevelDB 配置可能会不同。
>

### Repairer
当 LevelDB 异常退出时，可能出现 Manifest 写入错误或者 SSTable 写入错误。LevelDB 为用户提供了 Repairer（位于`db/repair.cc`）来尝试修复数据库文件。

Repairer 对外仅仅提供`Repairer::run`这个方法，我们看看他做了什么事：

+ 通过`FindFiles`方法找到 LevelDB 中所有文件，包括 LOG, MANIFEST 和 SSTable。
+ `ConvertLogFilesToTables`方法将检索全部 LOG 文件，将其生成 SSTable-0。
+ `ExtractMetaData`方法通过`ScanTable`扫描所有 SSTable 文件，生成其`FileMetaData`对象  ，以便恢复版本信息。
+ 最后通过`WriteDescriptor`重建 Manifest。

> Q：将所有 LOG 文件转换为 SSTable-0，那不是可能出现非常多重复的数据吗？
>
> A： 确实是这样，恢复的 LOG 数据可能已经保存在 SSTable 文件中了。LevelDB 的解决方案是通过 Compaction 逐渐压缩低层 Level 或者手动在 Repair 之后进行 CompactRange。除此之外 LevelDB 还提供了可能的优化方案：  
>
> 1. 计算总大小并选择适当的最大级别M
> 2. 按表中最大序列号排序表文件
> 3. 对每个表：如果与早期表重叠，放入级别0；否则放入级别M
>

## Cache
### 简介
目前 LevelDB 的核心内容已经了解完毕，现在来将一些细枝末节的知识点如 Cache，Iterator 等

我们回忆一下 LevelDB 是如何进行查询操作的：我们会先读取 SSTable 的 Footer，之后根据元信息再读取 SSTable 的 Index Block 再读取 Data Block。在这过程中，磁盘中读取 SSTable 的开销是很大的，所以我们可以通过 Cache 缓存 SSTable 的数据从而提高效率。LevelDB中使用的缓存从功能上可分为两种：

1. Table Cache
+ 缓存打开的 SSTable 文件句柄和相关元数据
+ 由 TableCache 类实现
+ 默认缓存大小为 max_open_files - 10 个文件句柄
2. Block Cache
+ 缓存数据块的内容
+ 使用 LRU 策略
+ 默认大小为 8MB
+ 由 LRUCache 类实现

在开始之前还得了解 ShardedCache 的概念，ShardedCache 中文为"分片 Cache"。多线程或多进程环境中，多个线程同时访问缓存时，单一的缓存池可能会导致内存访问冲突和线程竞争。通过将缓存分成多个分片（shards），每个线程可以访问不同的缓存分片，减少锁的竞争，提高并发性。  

### Cache 的应用
了解 LevelDB 中 Cache 的应用，我们首先宏观的看看读取一个 Key 是怎么利用到 Cache 的，再自底向上地介绍并分析LevelDB中 LRUHandle、HandleTable、LRUCache、ShardedLRUCache 的实现。

1. 首先`DBImpl::Get()`方法会调用`Version::Get()`在 SSTable 中查找 Key。

```cpp
Status DBImpl::Get(const ReadOptions& options, const Slice& key,
                   std::string* value) {
    // ...
    if (mem->Get(lkey, value, &s)) {
      // Done
    } else if (imm != nullptr && imm->Get(lkey, value, &s)) {
      // Done
    } else {
      s = current->Get(options, lkey, value, &stats);
      have_stat_update = true;
    }
    // ...
}
```

2. `Version::Get()`会通过`ForEachOverlapping()`方法遍历每一层的 SSTable 文件，找到 Key 所在的 SSTable。
3. 找到 SSTable 之后，将这个 SSTable 元信息传入 TableCache 中。
4. 之后由 TableCache 负责处理这个 SSTable 文件，读出数据。

> Q： 仔细研究代码我们会发现 LevelDB 遍历每一层 Level 找到 SSTable 文件后，将它的`file_number`传给 `TableCache::Get()`，但是这个方法内部又调用`FindTable()`通过`file_number`查找文件。它不是已经找到 SSTable 文件了吗？
>
> A：首先，`file_number`在 LevelDB 中是 SSTable 文件的唯一标识符，它在文件系统中的实际名字时通过这个`file_number`生成的。其次，虽然我们找到了包含 key 的 SSTable 文件，但是我们并不会直接打开文件读取，而是通过 TableCache 来管理已打开的 SSTable。TableCache 使用 file_number 作为缓存的 key，如果这个 SSTable 已经被打开过，就可以直接从缓存中获取，避免重复打开文件。
>

### LRUHandle
LRUHandle 是表示缓存对象的结构体

```cpp
struct LRUHandle {
  void* value;
  void (*deleter)(const Slice&, void* value);
  LRUHandle* next_hash;
  LRUHandle* next;
  LRUHandle* prev;
  size_t charge;  // TODO(opt): Only allow uint32_t?
  size_t key_length;
  bool in_cache;     // Whether entry is in the cache.
  uint32_t refs;     // References, including cache reference, if present.
  uint32_t hash;     // Hash of key(); used for fast sharding and comparisons
  char key_data[1];  // Beginning of key

  Slice key() const {
    // next is only equal to this if the LRU handle is the list head of an
    // empty list. List heads never have meaningful keys.
    assert(next != this);

    return Slice(key_data, key_length);
  }
};
```

我们着重关注三个 `LRUHandle` 类型的指针：`next`、`prev`和`next_hash`。`next`和`prev`用于维护 LRUCache 中的两个链表：in-use 链表和 lru 链表。`next_hash`是 HashTable 中的解决的拉链法的指针。

此外，还需要了解一下`void* value`和`void (*deleter)(const Slice&, void* value)`。`void*`表示任意类型的指针，它指向 Cache 管理的要缓存的数据，`(*deleter)(const Slice&, void* value)`是一个删除器函数。

+ `FindTable`方法会通过`file_number`找到 SSTable 文件，通过`Table::Open()`打开文件。此时就会将 SSTable 文件从磁盘调入内存。
+ `TableAndFile`用于管理缓存项，将打开的文件句柄`file`和构建好的 Table 对象`table`存入结构体中。
+ 最后将 TableAndFile 结构体的指针作为 value 插入 Cache 中，也就是前面提到的`void* value`

这种设计方式下，Cache 可以正确清理不同类型的资源，而不需要知道具体类型。每种类型的缓存项都可以注册自己的删除器。

```cpp
static void DeleteEntry(const Slice& key, void* value) {
  TableAndFile* tf = reinterpret_cast<TableAndFile*>(value);
  delete tf->table;
  delete tf->file;
  delete tf;
}

Status TableCache::FindTable(uint64_t file_number, uint64_t file_size,
                             Cache::Handle** handle) {
    // ...
    TableAndFile* tf = new TableAndFile;
    tf->file = file;
    tf->table = table;
    *handle = cache_->Insert(key, tf, 1, &DeleteEntry);
    // ...
}
```

### HandleTable
HandleTable 是 LevelDB 哈希表的具体实现，采用拉链法解决冲突。`length_`是哈希表中 slot 的数量，`elems_`是已使用的 slot 数量。

```cpp
class HandleTable{
private:
    uint32_t length_;
    uint32_t elems_;
    LRUHandle** list_;
};
```

HandleTable 的核心函数是`FindPointer`方法和`Resize`方法。

```cpp
LRUHandle** FindPointer(const Slice& key, uint32_t hash) {
    LRUHandle** ptr = &list_[hash & (length_ - 1)];
    while (*ptr != nullptr && ((*ptr)->hash != hash || key != (*ptr)->key())) {
        ptr = &(*ptr)->next_hash;
    }
    return ptr;
}
```

`FindPointer`方法根据 hash 和 key 查找 slot，如果查找失败则返回能够插入 slot 的指针。`Resize`方法是扩展哈希表的方法，该方法会倍增 solt 大小，并重新分配空间。在重新分配 solt 的空间后，再对所有原有 solt 中的 LRUHandle 重哈希。最后释放旧的 solt 的空间

```cpp
void Resize() {
    uint32_t new_length = 4;
    while (new_length < elems_) {
      new_length *= 2;
    }
    LRUHandle** new_list = new LRUHandle*[new_length];
    memset(new_list, 0, sizeof(new_list[0]) * new_length);
    uint32_t count = 0;
    for (uint32_t i = 0; i < length_; i++) {
      LRUHandle* h = list_[i];
      while (h != nullptr) {
        LRUHandle* next = h->next_hash;
        uint32_t hash = h->hash;
        LRUHandle** ptr = &new_list[hash & (new_length - 1)];
        h->next_hash = *ptr;
        *ptr = h;
        h = next;
        count++;
      }
    }
    assert(elems_ == count);
    delete[] list_;
    list_ = new_list;
    length_ = new_length;
  }
```

LevelDB 中哈希表保持桶的个数为 2 的次幂，从而使用位运算来通过键的哈希值快速计算出桶位置。每次插入若空间不足则会调用`Resize()`方法，将空间扩充到`new_length >= elems_`，并且重新哈希。

讲到 HandleTable 还需要了解 `in_use_`和`lru_`链表。

1. in-use 链表。所有正在被客户端使用的数据条目都存在该链表中，该链表是无序的，因为在容量不够时，此链表中的条目是一定不能够被驱逐的，因此也并不需要维持一个驱逐顺序。
2. lru 链表。所有已经不再为客户端使用的条目都放在 `lru_` 链表中，该链表按最近使用时间有序，当容量不够用时，会驱逐此链表中最久没有被使用的条目。

```cpp
void LRUCache::Ref(LRUHandle* e) {
  if (e->refs == 1 && e->in_cache) {  // If on lru_ list, move to in_use_ list.
    LRU_Remove(e);
    LRU_Append(&in_use_, e);
  }
  e->refs++;
}

void LRUCache::Unref(LRUHandle* e) {
  assert(e->refs > 0);
  e->refs--;
  if (e->refs == 0) {  // Deallocate.
    assert(!e->in_cache);
    (*e->deleter)(e->key(), e->value);
    free(e);
  } else if (e->in_cache && e->refs == 1) {
    // No longer in use; move to lru_ list.
    LRU_Remove(e);
    LRU_Append(&lru_, e);
  }
}
```

调用`LookUp()`方法查询 Key 时，会增加 LRUHandle 的引用计数，将 LRUHandle 插入`in_use_`链表。调用`Release()`方法释放 Key 时，会减少 LRUHandle 的引用计数，若`ref==1`则将 LRUHandle 从`in_use_`移入`lru_`等待淘汰。

> 设置两个链表的好处在于，能够更细致的管理 Cache 中的全部缓存项。例如`Prune()`方法手动释放缓存项时可以直接遍历`lru_`链表。
>

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1740572035359-83f3b18c-d151-4351-ba13-cd7453976e05.png)

### LRUCache
LRUCache 集成了 HandleTable、in-use 链表和 lru 链表，我们主要讲讲核心函数`LRUCache::Insert()`。

![](https://cdn.nlark.com/yuque/0/2025/png/49907638/1740639363752-96831bb1-42a7-493f-ab0a-c9c86834795c.png)

### ShardedLRUCache
前面我们说过，LevelDB 为了解决多个线程同时访问 Cache 冲突的情况，采用分片 Cache，提高了并行度。策略比较简洁， 利用 Key 哈希值的前 `kNumShardBits = 4` 个 bit 作为分片路由,可以支持 `kNumShards = 1 << kNumShardBits = 16`个分片。总体来说 ShardedCache 没有太多逻辑，更像一个 Wrapper，将 LRUCache 组合起来。

```cpp
Handle* ShardedCache::Insert(const Slice& key, void* value, size_t charge,
                 void (*deleter)(const Slice& key, void* value)) override {
    const uint32_t hash = HashSlice(key);
    return shard_[Shard(hash)].Insert(key, hash, value, charge, deleter);
}
```

### TableCache
终于我们到了最上层的 TableCache，我们从头到尾整理一下 Cache 在 LevelDB 中是如何应用的。

1. 首先我们通过`DBImpl::Get()`查询 Key。
2. 当 Key 不存在于 MemTable 和 Immutable MemTable 中时，就会调用 `Version::Get()`在当前版本的 SSTable 文件中搜索 Key。
3. `Version::Get()`通过 `ForEachOverlapping` 遍历每一层 SSTable，找到符合条件的 SSTable 文件后，调用`Version::Match()`记录查找统计信息（Seek Miss），并且在 SSTable 中查找 Key。
4. `Version::Match()`实际上调用`TableCache::Get()`查找 Key 。如果未缓存过这个 SSTable 文件，则会从磁盘调进内存，将元信息存入 Cache 中；如果缓存过 SSTable 文件，直接调用 `Table::InternalGet`方法，在 SSTable 内部查找 Key。

```cpp
Status TableCache::Get(const ReadOptions& options, uint64_t file_number,
                       uint64_t file_size, const Slice& k, void* arg,
                       void (*handle_result)(void*, const Slice&,
                                             const Slice&)) {
  Cache::Handle* handle = nullptr;
  Status s = FindTable(file_number, file_size, &handle);
  if (s.ok()) {
    Table* t = reinterpret_cast<TableAndFile*>(cache_->Value(handle))->table;
    s = t->InternalGet(options, k, arg, handle_result);
    cache_->Release(handle);
  }
  return s;
}

Status TableCache::FindTable(uint64_t file_number, uint64_t file_size,
                             Cache::Handle** handle) {
  Status s;
  char buf[sizeof(file_number)];
  EncodeFixed64(buf, file_number);
  Slice key(buf, sizeof(buf));
  *handle = cache_->Lookup(key);
  if (*handle == nullptr) {
    std::string fname = TableFileName(dbname_, file_number);
    RandomAccessFile* file = nullptr;
    Table* table = nullptr;
    s = env_->NewRandomAccessFile(fname, &file);
    if (!s.ok()) {
      std::string old_fname = SSTTableFileName(dbname_, file_number);
      if (env_->NewRandomAccessFile(old_fname, &file).ok()) {
        s = Status::OK();
      }
    }
    if (s.ok()) {
      s = Table::Open(options_, file, file_size, &table);
    }

    if (!s.ok()) {
      assert(table == nullptr);
      delete file;
      // We do not cache error results so that if the error is transient,
      // or somebody repairs the file, we recover automatically.
    } else {
      TableAndFile* tf = new TableAndFile;
      tf->file = file;
      tf->table = table;
      *handle = cache_->Insert(key, tf, 1, &DeleteEntry);
    }
  }
  return s;
}
```



## Iterator






# 优化方向
| 优化方法 | 性能提高 | 使用场景 | 参考文献 |
| --- | --- | --- | --- |
| 多线程 Compaction | 高（多核系统） | 高并发写入 | [https://github.com/facebook/rocksdb/wiki/Compaction](https://github.com/facebook/rocksdb/wiki/Compaction) |
| 改进文件选择算法 | 中等（减少 I/O） | 键分布不均匀 | [https://github.com/facebook/rocksdb/wiki/Compaction](https://github.com/facebook/rocksdb/wiki/Compaction) |
| 压缩优化 | 中等（节省空间） | 存储空间受限 | |
| 写放大 | 高 |  | |


## 优化 Compaction 过程
### 


# Q&A
1. <font style="color:black;">为什么 LevelDB 在内存中选择用 Skiplist 这个数据结构？
2. <font style="color:black;">为什么 LevelDB 在 Cache 中选择 HashTable 这个数据结构？

