Ref:
- 考前背诵
- [含动画和比较表格](https://developer.aliyun.com/article/941774)
- [含分类图](https://blog.csdn.net/u011412768/article/details/107394325)
- [含对应leetcode](https://zhuanlan.zhihu.com/p/77700835)
- [桶排序](https://segmentfault.com/a/1190000022767400)

![sorting mindmap](https://github.com/tinghe14/MLE-Learner/blob/1213040e837c71afb0811353729c2e036fbdb3ba/Coding%20Language/Basic%20Algorithm%20from%20Scrach/sorting%20mindmap.png)

Sorting Algorithm: bubble sort(冒泡排序), insertion sort(插入排序)，merge sort(归并排序)，quick sort(快速排序), topological sort(拓扑排序), heap sort(堆排序), bucket sort(桶排序)
- Time Complexity:
  - O(n^2): bubble sort{space: O(1)}, insertion sort{O(1)}
  - O(nlogn): heap sort{O(1)}, merge sort{O(n)}, quick sort{O(logn)}
  - O(n): bucket sort{O(n+k)}

![sorting table](https://github.com/tinghe14/MLE-Learner/blob/1213040e837c71afb0811353729c2e036fbdb3ba/Coding%20Language/Basic%20Algorithm%20from%20Scrach/sorting%20table.png)

bubble sort(冒泡排序)
- 比较相邻的元素，如果第一个比第二个大，就交换他们两个。
- 对每一对相邻元素做同样的工作，从开始第一对到结尾的最后一对。在这一点，最后的元素应该会是最大的数
- 针对所有的元素重复以上的步骤，除了最后一个。
- 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
~~~
def bubble_sort(num);
  for pass_num in range(len(nums)-1, 0, -1): #n-1趟
    flag = True #每一趟设置flag为true
    for i in range(pass_num):
      if nums[i] > nums[i+1]:
        nums[i], nums[i+1] = nums[i+1], nums[i]
        flah = False #有交换： flag设置为false 
    #每一趟结束后打的数会往后冒泡
    #flag为true, 说明到这趟已经没有交换 提前跳出循环 提高算法效率
    if flag:
      break
  return nums
      
~~~
- 无序表初始数据项的排列状况对冒泡排序没有影响，算法过程总需要 n-1 趟，随着趟数的增加，比对次数逐步从 n-1 减少到1，并包括可能发生的数据项交换。比对次数是1～n-1的累加，比对的时间复杂度是O(n²)。
- 关于交换次数，时间复杂度也是O(n2)，通常每次交换包括3次赋值。最好的情况是列表在排序前已经有序，交换次数为0；最差的情况是每次比对都要进行交换，交换次数等于比对次数，平均情况则是最差情况的一半。
- 冒泡排序通常作为时间效率较差的排序算法，来作为其它算法的对比基准。其效率主要差在每个数据项在找到其最终位置之前，必须要经过多次比对和交换，其中大部分的操作是无效的。
- 冒泡排序的优势在于无需任何额外的存储空间开销。

insertion sort(插入排序)
- 插入排序的代码实现虽然没有冒泡排序和选择排序那么简单粗暴，但它的原理应该是最容易理解的了，因为只要打过扑克牌的人都应该能够秒懂。插入排序是一种最简单直观的排序算法，它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序和冒泡排序一样，也有一种优化算法，叫做拆半插入
- 算法原理：
  - 将第一待排序序列第一个元素看做一个有序序列，把第二个元素到最后一个元素当成是未排序序列。
  - 从头到尾依次扫描未排序序列，将扫描到的每个元素插入有序序列的适当位置。（如果待插入的元素与有序序列中的某个元素相等，则将待插入元素插入到相等元素的后面）
~~~
def insert_sort(arr):
  for index_ in range(1, len(arr)):
    position = index_
    current_value = arr[inedx_] #插入项
    # 比对 移动 直到找到第一个比他小的项
    while position > 0 and arra[position-1] > current_value:
      arr[position] = arr[position-1]
      position = position-1
    #插入新项
    arr[position] = current_value
 return arr
~~~
- 插入排序的比对主要用来寻找 “新项” 的插入位置，最差情况是每趟都与子列表中所有项进行比对，总比对次数与冒泡排序相同，数量级仍是O(n²) 。
- 最好情况，列表已经排好序的时候，每趟仅需 1 次比对，总次数是O(n)。
- 由于移动操作仅包含1次赋值，是交换操作的1/3，所以插入排序性能会较好一些。

插入排序的更高效改进版本：希尔排序。但希尔排序是不稳定排序算法。希尔排序是基于插入排序的以下两点性质而提出改进方法的：
- 插入排序在对几乎已经排好序的数据操作时，效率高，即可以达到线性排序的效率；但插入排序一般来说是低效的，因为插入排序每次只能将数据移动一位；
- 希尔排序的基本思想是：先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，待整个序列中的记录“基本有序”时，再对全体记录进行依次直接插入排序。
- 算法原理：
  - 选择一个增量序列 t1，t2，……，tk，其中 ti > tj, tk = 1；
  - 按增量序列个数 k，对序列进行 k 趟排序；
  - 每趟排序，根据对应的增量 ti，将待排序列分割成若干长度为 m 的子序列，分别对各子表进行直接插入排序。仅增量因子为 1 时，整个序列作为一个表来处理，表长度即为整个序列的长度。
~~~
def shell_sort(nums):
  n = len(nums)
  gap = n//2 #定义增量
  # gap等于1的时候相当于最后一步是插入排序
  while gap >= 1:
    for j in range(gap, n):
      i = j 
      #增量的插入排序版本
      while (i-gap) >= 0:
        if nums[i] < nums[i-gap]:
          nums[i], nums[i-gap] = nums[i-gap], nums[i]
          i -= gap 
        else:
          break 
    gap // 2
  return nums
~~~
- 粗看上去，谢尔排序以插入排序为基础，可能并不会比插入排序好，但由于每趟都使得列表更加接近有序，过程会减少很多原先需要的“无效”比对
- 对希尔排序的详尽分析比较复杂，大致说是介于 O(n) 和 O(n²) 之间如果将间隔保持在2的k次方-1(1、3、5、7、15、31等等），谢尔排序的时间复杂度约为O(n^1.5)。

merge sort(归并排序)：
- 归并排序（Merge sort）是建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。
- 作为一种典型的分而治之思想的算法应用，归并排序的实现由两种方法：
  - 自上而下的递归（所有递归的方法都可以用迭代重写，所以就有了第 2 种方法）；
  - 自下而上的迭代；
- 和选择排序一样，归并排序的性能不受输入数据的影响，但表现比选择排序好的多，因为始终都是 O(nlogn) 的时间复杂度。代价是需要额外的内存空间。
- 算法原理：
  - 开辟内存空间，使其大小为两个已经排序序列之和，该空间用来存放合并后的序列；
  - 设定两个指针，最初位置分别为两个已经排序序列的起始位置；
  - 比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针到下一位置；
  - 重复步骤 3 直到某一指针达到序列尾；
  - 将另一序列剩下的所有元素直接复制到合并序列尾。
~~~
#不好，见我leetcode数据结构那本资料上的merge sort写法
from math import floor

def merge_sort(arr):
  if (len(arr) < 2):
    return arr 
   # 二分
   middle = floor(len(arr) / 2)
   left, right = arr[0:middle], arr[middle:]
   # 递归
   return merge(merge_sort(left), merge_sort(right))
   
 def merge(left, right)
  result = []
  # 分治
  while left and right:
    if left[0] <= right[0]:
      result.append(left.pop(0))
    else:
      result.append(rigit.pop(0))
  while left:
    result.append(left.pop(0))
  while right:
    result.append(right.pop(0))
  return result
~~~

quick sort(快速排序)
- 快速排序是由东尼·霍尔所发展的一种排序算法。在平均状况下，排序 n 个项目要 Ο(nlogn) 次比较。在最坏状况下则需要 Ο(n²) 次比较，但这种状况并不常见。事实上，快速排序通常明显比其他 Ο(nlogn) 算法更快，因为它的内部循环（inner loop）可以在大部分的架构上很有效率地被实现出来。
- 快速排序使用分治法（Divide and conquer）策略来把一个串行（list）分为两个子串行（sub-lists）。
- 快速排序又是一种分而治之思想在排序算法上的典型应用。本质上来看，快速排序应该算是在冒泡排序基础上的递归分治法。
- 快速排序的名字起的是简单粗暴，因为一听到这个名字你就知道它存在的意义，就是快，而且效率高！它是处理大数据最快的排序算法之一了。虽然 Worst Case 的时间复杂度达到了 O(n²)，但是人家就是优秀，在大多数情况下都比平均时间复杂度为 O(n logn) 的排序算法表现要更好。查阅资料了解到：快速排序的最坏运行情况是 O(n²)，比如说顺序数列的快排。但它的平摊期望时间是 O(nlogn)，且 O(nlogn) 记号中隐含的常数因子很小，比复杂度稳定等于 O(nlogn) 的归并排序要小很多。所以，对绝大多数顺序性较弱的随机数列而言，快速排序总是优于归并排序。
- 算法原理：
  - 从数列中挑出一个元素，称为 “基准”（pivot）;
  - 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
  - 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序；递归的最底部情形，是数列的大小是0或1，也就是永远都已经被排序好了。虽然一直递归下去，但是这个算法总会退出，因为在每次的迭代（iteration）中，它至少会把一个元素摆到它最后的位置去。
 ~~~
 def quick_sort(arr, left=None, right=None):
    left = 0 if not isinstance(left,(int, float)) else left
    right = len(arr) - 1 if not isinstance(right,(int, float)) else right
    if left < right:
        partitionIndex = partition(arr, left, right)
        quick_sort(arr, left, partitionIndex - 1)
        quick_sort(arr, partitionIndex + 1, right)
    return arr

def partition(arr, left, right):
    pivot = left
    index = pivot + 1
    i = index
    while  i <= right:
        if arr[i] < arr[pivot]:
            swap(arr, i, index)
            index += 1
        i+=1
    swap(arr, pivot, index - 1)
    return index - 1

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
~~~

heap sort(堆排序)
- 堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点。堆排序可以说是一种利用堆的概念来排序的选择排序。分为两种方法
- 大顶堆：每个节点的值都大于或等于其子节点的值，在堆排序算法中用于升序排列；
- 小顶堆：每个节点的值都小于或等于其子节点的值，在堆排序算法中用于降序排列；
堆排序的平均时间复杂度为 Ο(nlogn)。
- 算法原理：
  - 将待排序序列构建成一个堆 H[0……n-1]，根据（升序降序需求）选择大顶堆或小顶堆；
  - 把堆首（最大值）和堆尾互换；
  - 把堆的尺寸缩小 1，并调用 shift_down(0)，目的是把新的数组顶端数据调整到相应位置；
  - 重复步骤 2，直到堆的尺寸为1。
~~~
from math import floor

def buildMaxHeap(arr):
    for i in range(floor(len(arr)/2), -1, -1):
        heapify(arr, i)

def heapify(arr, i):
    left = 2 * i + 1
    right = 2 * i + 2
    largest = i
    if left < arrLen and arr[left] > arr[largest]:
        largest = left
    if right < arrLen and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        swap(arr, i, largest)
        heapify(arr, largest)

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def heap_sort(arr):
    global arrLen
    arrLen = len(arr)
    buildMaxHeap(arr)
    for i in range(len(arr)-1, 0, -1):
        swap(arr, 0, i)
        arrLen -= 1
        heapify(arr, 0)
    return arr
~~~

bucket sort(桶排序)
- 也叫箱排序，其主要思想是：将待排序集合中处于同一个值域的元素存入同一个桶中，也就是根据元素值特性将集合拆分为多个区域，则拆分后形成的多个桶，从值域上看是处于有序状态的。对每个桶中元素进行排序，则所有桶中元素构成的集合是已排序的。桶排序是计数排序的扩展版本，计数排序可以看成每个桶只存储相同元素，而桶排序每个桶存储一定范围的元素。桶排序需要尽量保证元素分散均匀，否则当所有数据集中在同一个桶中时，桶排序失效。
- 算法实现：
  - 根据待排序集合中最大元素和最小元素的差值范围和映射规则，确定申请的桶个数；
  - 遍历排序序列，将每个元素放到对应的桶里去；
  - 对不是空的桶进行排序；
  - 按顺序访问桶，将桶中的元素依次放回到原序列中对应的位置，完成排序。
~~~
# bucket_sort 代码实现

from typing import List

def bucket_sort(arr:List[int]):
    """桶排序"""
    min_num = min(arr)
    max_num = max(arr)
    # 桶的大小
    bucket_range = (max_num-min_num) / len(arr)
    # 桶数组
    count_list = [ [] for i in range(len(arr) + 1)]
    # 向桶数组填数
    for i in arr:
        count_list[int((i-min_num)//bucket_range)].append(i)
    arr.clear()
    # 回填，这里桶内部排序直接调用了sorted
    for i in count_list:
        for j in sorted(i):
            arr.append(j)
~~~
