/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include "kmeans_trainer.hpp"

bob::learn::em::KMeansTrainer::KMeansTrainer(InitializationMethod i_m)
  : m_rng(new std::mt19937()),
    m_average_min_distance(0)
{
  m_initialization_method = i_m;
}


bob::learn::em::KMeansTrainer::KMeansTrainer(const bob::learn::em::KMeansTrainer& other)
{
  m_initialization_method = other.m_initialization_method;
  m_rng                   = other.m_rng;
  m_average_min_distance  = other.m_average_min_distance;
  m_zeroethOrderStats     = other.m_zeroethOrderStats;
  m_firstOrderStats       = other.m_firstOrderStats;
}


bob::learn::em::KMeansTrainer& bob::learn::em::KMeansTrainer::operator=
(const bob::learn::em::KMeansTrainer& other)
{
  if(this != &other)
  {
    m_rng                         = other.m_rng;
    m_initialization_method       = other.m_initialization_method;
    m_average_min_distance        = other.m_average_min_distance;

    m_zeroethOrderStats = other.m_zeroethOrderStats;
    m_firstOrderStats   = other.m_firstOrderStats;
  }
  return *this;
}


bool bob::learn::em::KMeansTrainer::operator==(const bob::learn::em::KMeansTrainer& b) const {
  return m_initialization_method == b.m_initialization_method &&
         *m_rng == *(b.m_rng) && m_average_min_distance == b.m_average_min_distance &&
         // bob::core::array::hasSameShape(m_zeroethOrderStats, b.m_zeroethOrderStats) &&
         // bob::core::array::hasSameShape(m_firstOrderStats, b.m_firstOrderStats) &&
         m_zeroethOrderStats == b.m_zeroethOrderStats &&
         m_firstOrderStats == b.m_firstOrderStats;
}

bool bob::learn::em::KMeansTrainer::operator!=(const bob::learn::em::KMeansTrainer& b) const {
  return !(this->operator==(b));
}

template <class E>
void bob::learn::em::KMeansTrainer::initialize(
  bob::learn::em::KMeansMachine& kmeans,
  const xt::xexpression<E>& ar_expr)
{
  const auto& ar = ar_expr.derived_cast();
  // split data into as many chunks as there are means
  size_t n_data = ar.shape()[0];

  // assign the i'th mean to a random example within the i'th chunk
  // blitz::Range a = blitz::Range::all();
  if(m_initialization_method == RANDOM || m_initialization_method == RANDOM_NO_DUPLICATE) // Random initialization
  {
    unsigned int n_chunk = n_data / kmeans.getNMeans();
    size_t n_max_trials = (size_t) n_chunk * 5;
    xt::xtensor<double, 1> cur_mean;

    if(m_initialization_method == RANDOM_NO_DUPLICATE)
      cur_mean.reshape({kmeans.getNInputs()});

    for(size_t i = 0; i < kmeans.getNMeans(); ++i)
    {
      std::uniform_int_distribution<> die(i * n_chunk, (i + 1) * n_chunk - 1);

      // get random index within chunk
      unsigned int index = die(*m_rng);

      // get the example at that index
      xt::xtensor<double, 1> mean = xt::view(ar, index, xt::all());

      if(m_initialization_method == RANDOM_NO_DUPLICATE)
      {
        size_t count = 0;
        while(count < n_max_trials)
        {
          // check that the selected sampled is different than all the previously
          // selected ones
          bool valid = true;
          for(size_t j = 0; j < i && valid; ++j)
          {
            cur_mean = kmeans.getMean(j);
            valid = mean != cur_mean;
          }
          // if different, stop otherwise, try with another one
          if(valid)
            break;
          else
          {
            index = die(*m_rng);
            mean = xt::view(ar, index, xt::all());
            ++count;
          }
        }
        // Initialization fails
        if(count >= n_max_trials) {
          throw std::runtime_error("initialization failure: surpassed the maximum number of trials "s + std::to_string(n_max_trials));
        }
      }

      // set the mean
      kmeans.setMean(i, mean);
    }
  }
  else // K-Means++
  {
    // 1.a. Selects one sample randomly
    std::uniform_int_distribution<> die(0, n_data-1);
    //   Gets the example at a random index
    xt::xtensor<double, 1> mean = xt::view(ar, die(*m_rng), xt::all());
    kmeans.setMean(0, mean);

    // 1.b. Loops, computes probability distribution and select samples accordingly
    auto weights = xt::xtensor<double, 1>::from_shape({n_data});
    for(size_t m = 1; m < kmeans.getNMeans(); ++m)
    {
      // For each sample, puts the distance to the closest mean in the weight vector
      for(size_t s=0; s < n_data; ++s)
      {
        xt::xtensor<double, 1> s_cur = xt::view(ar, s, xt::all());
        double& w_cur = weights(s);
        // Initializes with the distance to first mean
        w_cur = kmeans.getDistanceFromMean(s_cur, 0);
        // Loops over the remaining mean and update the mean distance if required
        for(size_t i = 1; i < m; ++i)
          w_cur = std::min(w_cur, kmeans.getDistanceFromMean(s_cur, i));
      }
      // Square and normalize the weights vectors such that
      // \f$weights[x] = D(x)^{2} \sum_{y} D(y)^{2}\f$
      weights = xt::pow(weights, 2);
      weights /= xt::sum(weights);

      // Takes a sample according to the weights distribution
      // Blitz iterators is fine as the weights array should be C-style contiguous
      // bob::core::array::assertCContiguous(weights);
      // 
      // TODO fix up this
      std::discrete_distribution<> die2(weights.begin(), weights.end());
      xt::xtensor<double, 1> new_mean = xt::view(ar, die2(*m_rng), xt::all());
      kmeans.setMean(m, new_mean);
    }
  }
}

template <class E>
void bob::learn::em::KMeansTrainer::eStep(bob::learn::em::KMeansMachine& kmeans,
  const xt::xexpression<E>& ar_expr)
{
  const auto& ar = ar_expr.derived_cast();
  // initialise the accumulators
  resetAccumulators(kmeans);

  // iterate over data samples
  // blitz::Range a = blitz::Range::all();
  for(int i = 0; i < ar.shape()[0]; ++i) {
    // get example
    xt::xtensor<double, 1> x(xt::view(ar, i, xt::all()));

    // find closest mean, and distance from that mean
    size_t closest_mean = 0;
    double min_distance = 0;
    kmeans.getClosestMean(x, closest_mean, min_distance);

    // accumulate the stats
    m_average_min_distance += min_distance;
    ++m_zeroethOrderStats(closest_mean);
    xt::view(m_firstOrderStats, closest_mean, xt::all()) += x;
  }
  m_average_min_distance /= static_cast<double>(ar.shape()[0]);
}

template <class E>
void bob::learn::em::KMeansTrainer::mStep(bob::learn::em::KMeansMachine& kmeans,
  const xt::xexpression<E>& /*unused*/)
{
  xt::xtensor<double, 2>& means = kmeans.updateMeans();
  for(size_t i = 0; i < kmeans.getNMeans(); ++i)
  {
    xt::view(means, i, xt::all()) = xt::view(m_firstOrderStats, i, xt::all()) / m_zeroethOrderStats(i);
  }
}

double bob::learn::em::KMeansTrainer::computeLikelihood(bob::learn::em::KMeansMachine& kmeans)
{
  return m_average_min_distance;
}


void bob::learn::em::KMeansTrainer::resetAccumulators(bob::learn::em::KMeansMachine& kmeans)
{
   // Resize the accumulator
  m_zeroethOrderStats.reshape({kmeans.getNMeans()});
  m_firstOrderStats.reshape({kmeans.getNMeans(), kmeans.getNInputs()});

  // initialize with 0
  m_average_min_distance = 0;
  std::fill(m_zeroethOrderStats.begin(), m_zeroethOrderStats.end(), 0.);
  std::fill(m_firstOrderStats.begin(), m_firstOrderStats.end(), 0.);
}

void bob::learn::em::KMeansTrainer::setZeroethOrderStats(const xt::xtensor<double, 1>& zeroethOrderStats)
{
  // bob::core::array::assertSameShape(m_zeroethOrderStats, zeroethOrderStats);
  m_zeroethOrderStats = zeroethOrderStats;
}

void bob::learn::em::KMeansTrainer::setFirstOrderStats(const xt::xtensor<double,2>& firstOrderStats)
{
  // bob::core::array::assertSameShape(m_firstOrderStats, firstOrderStats);
  m_firstOrderStats = firstOrderStats;
}
