# MDN-test

test of inferpy, MDN, and an issue I dont know how to fix

Issue that occured : why do the posterior_predictive and VI.fit need same amout of data. 
For now they havent fixed this issue, but if the size is not the same, add random numbers to test:

```
padded_x_test = x_train
padded_x_test[:x_test.shape[0],:x_test.shape[1]]=x_test

y_pred_list = []
for i in range(1000):
    y_test_pred = m.posterior_predictive(["y"], data = {"x": padded_x_test}).sample()[:x_test.shape[0]]
    y_pred_list.append(y_test_pred)
```


Other than that, it is a pretty cool example, in a subject I need to learn more about!

