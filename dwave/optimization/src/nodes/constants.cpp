// Copyright 2024 D-Wave
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#include "dwave-optimization/nodes/constants.hpp"

#include <mutex>
#include <ranges>

#include "dwave-optimization/utils.hpp"
#include "_state.hpp"

namespace dwave::optimization {

// We're a bit memory-sensitive. So rather than saving a mutex on each instance
// of the class, we just make a global one on the assumption that there will
// only ever be one cache miss per node.
std::mutex buffer_stats_mutex;

ConstantNode::BufferStats::BufferStats(std::span<const double> buffer)
        : integral(std::ranges::all_of(buffer, is_integer)),
          min(std::ranges::min(buffer)),
          max(std::ranges::max(buffer)) {
    assert(!buffer.empty() && "buffer cannot be empty");
}

bool ConstantNode::integral() const {
    auto values = this->data();  // all of the values in the array

    // If we're empty we return true because this gives maximum compatibility
    // with other nodes. There are several other nodes that require integral,
    // and none (as of Sept 2024) that require *not* integral.
    if (values.empty()) return true;

    // If we're a scalar (or a size-1 array) we can just calculate it in O(1) so let's
    // not bother caching.
    if (values.size() == 1) return is_integer(values[0]);

    // Construct the cache if it's not already there.
    // This only ever happens once, so we do one check outside the mutex for
    // speed, and then another within is to make sure someone else hasn't
    // already constructed it. Subsequent reads are safe
    if (!buffer_stats_) {
        std::lock_guard<std::mutex> guard(buffer_stats_mutex);
        if (!buffer_stats_) buffer_stats_.emplace(values);
    }

    // Return the cached value
    return buffer_stats_->integral;
}

double ConstantNode::max() const {
    auto values = this->data();  // all of the values in the array

    // If we're empty we return 0.
    // We don't want undefined behavior because there are use cases e.g.
    // indexing by an empty array.
    // So 0 seems like a reasonable default.
    if (values.empty()) return 0.0;

    // If we're a scalar (or a size-1 array) we can just calculate it in O(1) so let's
    // not bother caching.
    if (values.size() == 1) return values[0];

    // Construct the cache if it's not already there.
    // This only ever happens once, so we do one check outside the mutex for
    // speed, and then another within is to make sure someone else hasn't
    // already constructed it. Subsequent reads are safe
    if (!buffer_stats_) {
        std::lock_guard<std::mutex> guard(buffer_stats_mutex);
        if (!buffer_stats_) buffer_stats_.emplace(values);
    }

    // Return the cached value
    return buffer_stats_->max;
}

double ConstantNode::min() const {
    auto values = this->data();  // all of the values in the array

    // If we're empty we return 0.
    // We don't want undefined behavior because there are use cases e.g.
    // indexing by an empty array.
    // So 0 seems like a reasonable default.
    if (values.empty()) return 0.0;

    // If we're a scalar (or a size-1 array) we can just calculate it in O(1) so let's
    // not bother caching.
    if (values.size() == 1) return values[0];

    // Construct the cache if it's not already there.
    // This only ever happens once, so we do one check outside the mutex for
    // speed, and then another within is to make sure someone else hasn't
    // already constructed it. Subsequent reads are safe
    if (!buffer_stats_) {
        std::lock_guard<std::mutex> guard(buffer_stats_mutex);
        if (!buffer_stats_) buffer_stats_.emplace(values);
    }

    // Return the cached value
    return buffer_stats_->min;
}

void InputNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    std::vector<double> values;
    values.assign(this->data().begin(), this->data().end());
    state[index] = std::make_unique<ArrayNodeStateData>(std::move(values));
}

double const* InputNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

std::span<const Update> InputNode::diff(const State& state) const noexcept {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void InputNode::commit(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

void InputNode::revert(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

void InputNode::assign(State& state, std::span<const double> new_values) const {
    if (static_cast<ssize_t>(new_values.size()) != this->size()) {
        throw std::invalid_argument("size of new values must match");
    }

    BufferStats stats(new_values);
    if (stats.min < min()) {
        throw std::invalid_argument("new data contains a value smaller than the min");
    }
    if (stats.max > max()) {
        throw std::invalid_argument("new data contains a value smaller than the min");
    }
    if (integral() && !stats.integral) {
        throw std::invalid_argument("new data contains a non-integral value");
    }

    data_ptr<ArrayNodeStateData>(state)->assign(new_values);
}

void InputNode::assign(State& state, const std::vector<double>& new_values) const {
    this->assign(state, std::span(new_values));
}

}  // namespace dwave::optimization
