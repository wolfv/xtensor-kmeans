/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include "kmeans_machine.hpp"
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xview.hpp>
// #include <bob.core/array_copy.h>
#include <limits>

using namespace std::string_literals;

bob::learn::em::KMeansMachine::KMeansMachine()

  // : m_n_means(0), m_n_inputs(0), m_means(0,0),
  //   m_cache_means(std::array<std::size_t,>{0,0})
{
  // m_means = 0;
}

bob::learn::em::KMeansMachine::KMeansMachine(const size_t n_means, const size_t n_inputs):
  m_n_means(n_means),
  m_n_inputs(n_inputs),
  m_means(std::array<std::size_t, 2>{n_means, n_inputs}),
  m_cache_means(std::array<std::size_t, 2>{n_means, n_inputs})
{
  // m_means = 0;
}

bob::learn::em::KMeansMachine::KMeansMachine(const xt::xtensor<double, 2>& means):
  m_n_means(means.shape()[0]), m_n_inputs(means.shape()[1]),
  m_means(means),
  m_cache_means(means.shape())
{
}

bob::learn::em::KMeansMachine::KMeansMachine(const bob::learn::em::KMeansMachine& other):
  m_n_means(other.m_n_means), m_n_inputs(other.m_n_inputs),
  m_means(other.m_means),
  m_cache_means(other.m_cache_means.shape())
{
}

// bob::learn::em::KMeansMachine::KMeansMachine(bob::io::base::HDF5File& config)
// {
//   load(config);
// }

bob::learn::em::KMeansMachine::~KMeansMachine() { }

bob::learn::em::KMeansMachine& bob::learn::em::KMeansMachine::operator=
(const bob::learn::em::KMeansMachine& other)
{
  if(this != &other)
  {
    m_n_means = other.m_n_means;
    m_n_inputs = other.m_n_inputs;
    // m_means.reference(bob::core::array::ccopy(other.m_means));
    m_means = other.m_means;
    m_cache_means.reshape(other.m_means.shape());
  }
  return *this;
}

bool bob::learn::em::KMeansMachine::operator==(const bob::learn::em::KMeansMachine& b) const
{
  return m_n_inputs == b.m_n_inputs && m_n_means == b.m_n_means &&
         xt::all(xt::equal(m_means, b.m_means));
}

bool bob::learn::em::KMeansMachine::operator!=(const bob::learn::em::KMeansMachine& b) const
{
  return !(this->operator==(b));
}

bool bob::learn::em::KMeansMachine::is_similar_to(const bob::learn::em::KMeansMachine& b,
  const double r_epsilon, const double a_epsilon) const
{
  return m_n_inputs == b.m_n_inputs && m_n_means == b.m_n_means &&
         xt::all(xt::isclose(m_means, b.m_means, r_epsilon, a_epsilon));
}

// void bob::learn::em::KMeansMachine::load(bob::io::base::HDF5File& config)
// {
  //reads all data directly into the member variables
  // m_means.reference(config.readArray<double, 2>("means"));
  // m_n_means = m_means.extent(0);
  // m_n_inputs = m_means.extent(1);
  // m_cache_means.resize(m_n_means, m_n_inputs);
// }

// void bob::learn::em::KMeansMachine::save(bob::io::base::HDF5File& config) const
// {
  // config.setArray("means", m_means);
// }

void bob::learn::em::KMeansMachine::setMeans(const xt::xtensor<double, 2>& means)
{
  m_means = means;
}

void bob::learn::em::KMeansMachine::setMean(const size_t i, const xt::xtensor<double, 1>& mean)
{
  if(i >= m_n_means) {
    // boost::format m("cannot set mean with index %lu: out of bounds [0,%lu[");
    // m % i % m_n_means;
    throw std::runtime_error("cannot set mean with index "s + std::to_string(i) + ": out of bounds [0, " + std::to_string(m_n_means) + "]");
  }
  xt::view(m_means, i, xt::all()) = mean;
}

const xt::xtensor<double, 1> bob::learn::em::KMeansMachine::getMean(const size_t i) const
{
  if(i>=m_n_means) {
    // boost::format m("cannot get mean with index %lu: out of bounds [0,%lu[");
    // m % i % m_n_means;
    throw std::runtime_error("cannot get mean with index"s + std::to_string(i));
  }

  return xt::view(m_means, i, xt::all());

}

double bob::learn::em::KMeansMachine::getDistanceFromMean(
  const xt::xtensor<double, 1>& x,
  const size_t i) const
{
  return xt::sum(xt::pow(xt::view(m_means, i, xt::all()) - x, 2))();
}

void bob::learn::em::KMeansMachine::getClosestMean(
  const xt::xtensor<double, 1>& x,
  size_t& closest_mean,
  double& min_distance) const
{
  min_distance = std::numeric_limits<double>::max();

  for(size_t i = 0; i < m_n_means; ++i) {
    double this_distance = getDistanceFromMean(x, i);
    if(this_distance < min_distance) {
      min_distance = this_distance;
      closest_mean = i;
    }
  }
}

double bob::learn::em::KMeansMachine::getMinDistance(const xt::xtensor<double, 1>& input) const
{
  size_t closest_mean = 0;
  double min_distance = 0;
  getClosestMean(input,closest_mean,min_distance);
  return min_distance;
}

void bob::learn::em::KMeansMachine::getVariancesAndWeightsForEachClusterInit(
  xt::xtensor<double, 2>& variances,
  xt::xtensor<double, 1>& weights) const
{
  // check arguments
  // initialise output arrays  variances = 0;
  std::fill(weights.begin(), weights.end(), 0.);
  std::fill(m_cache_means.begin(), m_cache_means.end(), 0.);
}

void bob::learn::em::KMeansMachine::getVariancesAndWeightsForEachClusterAcc(
    const xt::xtensor<double, 2>& data, 
    xt::xtensor<double, 2>& variances, 
    xt::xtensor<double, 1>& weights) const
{
  // check arguments
  // iterate over data
  // blitz::Range a = blitz::Range::all();
  for(int i = 0; i < data.shape()[0]; ++i) {
    // - get example
    xt::xtensor<double, 1> x(xt::view(data, i, xt::all()));

    // - find closest mean
    size_t closest_mean = 0;
    double min_distance = 0;
    getClosestMean(x, closest_mean, min_distance);

    // - accumulate stats
    xt::view(m_cache_means, closest_mean, xt::all()) += x;
    xt::view(variances, closest_mean, xt::all()) += xt::pow(x, 2);
    ++weights(closest_mean);
  }
}

void bob::learn::em::KMeansMachine::getVariancesAndWeightsForEachClusterFin(
  xt::xtensor<double, 2>& variances,
  xt::xtensor<double, 1>& weights) const
{
  // check arguments
  // calculate final variances and weights
  // blitz::firstIndex idx1;
  // blitz::secondIndex idx2;
  // TODO convert firstIndex / secondIndex!!
  // find means
  for (std::size_t i = 0; i < m_cache_means.shape()[0]; ++i)
  {
    for (std::size_t j = 0; j < m_cache_means.shape()[1]; ++j)
    {
      m_cache_means(i, j) = m_cache_means(i, j) / weights(i);
      variances(i, j) = variances(i, j) / weights(i);
    }
  }
  // m_cache_means = m_cache_means(idx1, idx2) / weights(idx1);

  // find variances
  variances -= xt::pow(m_cache_means, 2);

  // find weights
  weights = weights / xt::sum(weights);
}

void bob::learn::em::KMeansMachine::setCacheMeans(const xt::xtensor<double, 2>& cache_means)
{
  m_cache_means = cache_means;
}

void bob::learn::em::KMeansMachine::getVariancesAndWeightsForEachCluster(
  const xt::xtensor<double, 2>& data,
  xt::xtensor<double, 2>& variances,
  xt::xtensor<double, 1>& weights) const
{
  // initialise
  getVariancesAndWeightsForEachClusterInit(variances, weights);
  // accumulate
  getVariancesAndWeightsForEachClusterAcc(data, variances, weights);
  // merge/finalize
  getVariancesAndWeightsForEachClusterFin(variances, weights);
}

void bob::learn::em::KMeansMachine::forward(const xt::xtensor<double, 1>& input, double& output) const
{
  if(static_cast<size_t>(input.shape()[0]) != m_n_inputs) {
    std::cout << "Machine input size " << m_n_inputs << " does not match the size of input array " << input.shape()[0] << std::endl;
    throw std::runtime_error("blabla");
  }
  forward_(input, output);
}

void bob::learn::em::KMeansMachine::forward_(const xt::xtensor<double, 1>& input, double& output) const
{
  output = getMinDistance(input);
}

void bob::learn::em::KMeansMachine::resize(const size_t n_means, const size_t n_inputs)
{
  m_n_means = n_means;
  m_n_inputs = n_inputs;
  m_means.reshape({n_means, n_inputs});
  m_cache_means.reshape({n_means, n_inputs});
}

namespace bob { namespace learn { namespace em {
  std::ostream& operator<<(std::ostream& os, const KMeansMachine& km) {
    os << "Means = " << km.m_means << std::endl;
    return os;
  }
} } }
