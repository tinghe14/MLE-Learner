Defensive Coding:
- improve code quality and make code safer
- ref: https://github.com/ictar/python-doc/blob/master/Others/%E5%9C%A8Python%E4%B8%AD%E8%BF%9B%E8%A1%8C%E9%98%B2%E5%BE%A1%E6%80%A7%E7%BC%96%E7%A8%8B.md
- ref: https://developer.aliyun.com/article/950051

  Logical guidelines to write application code:
  1. guard clauses: Conduct precondition check of input variable at the top of your code making sure the methods only continue executing when valid input is provided
    - 不要相信用户的输入，对输入进行必要的检查
      - 数据格式是否正确
      - 数据类型是否正确
      - 数据长度是否正确
      - 可以按照如下思路来思考：
        - 如果参数缺失会有默认值吗
        - 如果参数问题，业务逻辑会有什么不合理的情况
        - 字段确实，不合法的情况，对于写的操作，是否会造成垃圾数据的产生
        - 结合业务场景来评估可能的影响范围
        - 必要的时候，设置白名单而不是黑名单
  3.  assertions inside methods: use assertive programming to verify assumptions as they write code. 断言简要的描述应用在运行时的期望状态，在更靠近问题根源的地方放置警告。断言可以用来处理代码中不应该反升的错误
  ~~~
  def normalize_ranges(colname):
    """ Normalize given data range to values in [0 - 1]
    Return dictionary new 'min' and 'max' keys in range [0 - 1]
    """
    # 1-D numpy array of data we loaded application with
    original_range = get_base_range(colname)
    colspan = original_range['datamax'] - original_range['datamin']
    # User filtered data from GUI
    live_data = get_column_data(colname)
    live_min = numpy.min(live_data)
    live_max = numpy.max(live_data)
    ratio = {}
    ratio['min'] = (live_min - original_range['datamin']) / colspan
    ratio['max'] = (live_max - original_range['datamin']) / colspan
    assert 0.0 <= ratio['min'] <= 1.0, ( # 返回布尔值
            '"%s" min (%f) not in [0-1] given (%f) colspan (%f)' % (
            colname, ratio['min'], original_range['datamin'], colspan)) # false时输出的失败消息信息
    assert 0.0 <= ratio['max'] <= 1.0, (
            '"%s" max (%f) not in [0-1] given (%f) colspan (%f)' % (
            colname, ratio['max'], original_range['datamax'], colspan))
    return ratio
  ~~~
  3.异常和错误处理，处理那些预料中可能要发生的错误。高级语言中一般用try catch的方式捕获异常
  ~~~
  try {
     //逻辑代码
  } catch (exception e){
     //异常处理代码
  }
   
  try{
     //逻辑代码
  } finally {
     //一定要执行的代码
  }
  
  try {
      //逻辑代码
  } catch (exception e){
      //异常处理代码
  } finally{
      //一定要执行的代码
  }
  ~~~
  4. logging
  5. unit test: not forget past bugs
  6. 缺点：非常容易过度使用，让代码非常嘈杂，掩埋了真正的功能。需要谨慎的使用，在哪些你假设永远不会发生的情况下使用，不要过分的来检查无效输入
