> *About this repo*
> This repo is a port of the kmeans-algorithm contained in the bob framework 
> by idiap. The dependency of blitz++ is removed, and xtensor is used in place. 
> In this document, the key differences and changes of the port are documented.
> Check out xtensor here: https://github.com/QuantStack/xtensor
> Original k-means source can be found here: https://gitlab.idiap.ch/bob/bob.learn.em

# Porting blitz++ to xtensor

Porting blitz++ code to xtensor is not very hard. Blitz++ and xtensor both support n-dimensional arrays,
which makes porting a worthwhile endeavour. 

### Arrays

In blitz++, the array type is called `blitz::Array<double, 2>` where `double` is the element type, and 
`2` is the number of dimensions.
The equivalent xtensor construct is the `xt::xtensor<double, 2>`.
However, xtensor also supports a datastructure with dynamic dimensions, called `xt::xarray<double>`.
The `xt::xarray` will dynamically allocate a shape and strides object (therefore it's slightly costlier to instantiate, and harder for the compiler to unroll certain loops etc.).

### Views

Another difference is how slicing and views work in xtensor vs. blitz. 

In blitz++ you can use the following code to create a view into an array:

```cpp
auto new = arr(blitz::Range::all(), blitz::Range(3, 5, 2));
arr(blitz::Range(2, 4)) = 1;
```

the equivalent in xtensor looks as follows:

```cpp
auto view = xt::view(arr, xt::all(), xt::range(3, 5, 2));
xt::view(arr, xt::all(), xt::range(3, 5, 2)) = 1;
```

i.e. we currently do not overload the `operator()` in order to create views.

### Index Placeholders

Currently, xtensor has no index placeholders. Therefore, the only way to 
create equivalent code is by generating for loops of the correct size
by hand. We might create a similar mechanism to indexplaceholders in the 
future (help always welcome!).

Some use cases can definitly be solved by broadcasting and use of `xt::arange`.

For example, `xt::arange(10)` creates `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`.
And `xt::broadcast(xt::arange(10), {3, 10})` would create an array of 

```cpp
{
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
}
```

Also, broadcasting with `xt::newaxis` can be particularly powerful. For example, 

```cpp
xt::xarray<double> a = xt::arange(3);

auto res = xt::view(a, xt::all(), xt::newaxis()) * a
// --> {{0, 0, 0},
        {0, 1, 2},
        {0, 2, 4}}
```
