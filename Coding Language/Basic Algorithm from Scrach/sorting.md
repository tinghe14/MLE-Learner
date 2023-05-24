Ref:
- 考前背诵
- [含动画和比较表格]（https://developer.aliyun.com/article/941774）
- [含分类图]（https://blog.csdn.net/u011412768/article/details/107394325）
- [含对应leetcode](https://zhuanlan.zhihu.com/p/77700835)

Sorting Algorithm: bubble sort(冒泡排序), insertion sort(插入排序)，merge sort(归并排序)，quick sort(快速排序), topological sort(拓扑排序), heap sort(堆排序), bucket sort(桶排序)
- Time Complexity:
  - O(n^2): bubble sort{space: O(1)}, insertion sort{O(1)}
  - O(nlogn): heap sort{O(1)}, merge sort{O(n)}, quick sort{O(logn)}
  - O(n): bucket sort{O(n+k)}

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
