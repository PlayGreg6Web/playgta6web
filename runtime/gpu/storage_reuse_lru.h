// Least recently used order of the relocated storage homes, with an index for the miss path's reuse
// scan (Device::relocate_storage_buffer, #663 / #2256).
//
// The scan walks the homes oldest first, at most `window` of them, stops at the first home still in
// use (bound fewer than `reuse_after` presents ago) and takes the first settled home of the same kind
// (dedicated or shared chunk slot) whose capacity is at least `need`. find_reuse_linear() is that walk.
// find_reuse() gives the same answer without the walk:
// - every node carries a sequence number that grows along the order (a bind moves the node to the back
//   with a new one), so "before" is a number compare;
// - `boundary_` is the node at rank `window` (the first one the walk cannot reach), kept in O(1) per
//   insert / erase / bind;
// - nodes are also listed per bind present (oldest first): the walk stops at the oldest node of the
//   presents that are not settled yet (a handful, the last reuse_after ones);
// - and per kind and capacity bucket (four buckets per power of two, oldest first): every node of a
//   bucket above need's bucket is large enough, so its front is its oldest candidate; need's own bucket
//   is walked from the front until a large enough node or a node past the best one so far.
// The answer is the oldest candidate if it is before both the boundary and the first unsettled node.
// Nodes are owned by the caller (embedded in its entries) and linked in place: no allocation per bind.
// Not thread-safe: the owner locks.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>

namespace playgta6::gpu::d3d12_guest {

template <class Entry>
class StorageReuseLru {
public:
    struct Node {
        Entry* entry = nullptr;
        uint64_t last_used_present = 0;  // present at the last bind (reuse guard, #663)
        uint64_t slot_capacity = 0;
        bool dedicated = false;

    private:
        friend class StorageReuseLru;
        struct PresentGroup;
        uint64_t seq_ = 0;
        Node* prev_ = nullptr;   // whole order
        Node* next_ = nullptr;
        Node* bprev_ = nullptr;  // capacity bucket
        Node* bnext_ = nullptr;
        Node* gprev_ = nullptr;  // present group
        Node* gnext_ = nullptr;
        PresentGroup* group_ = nullptr;
        struct PresentGroup {
            Node* head = nullptr;
            Node* tail = nullptr;
        };
    };

    explicit StorageReuseLru(size_t window) : window_(window) {}
    StorageReuseLru(const StorageReuseLru&) = delete;
    StorageReuseLru& operator=(const StorageReuseLru&) = delete;

    bool empty() const noexcept { return head_ == nullptr; }
    size_t size() const noexcept { return size_; }
    Node* front() const noexcept { return head_; }
    static Node* next(const Node* n) noexcept { return n->next_; }

    // Links `n` (not linked) at the back as bound at `present`.
    void push_back(Node* n, Entry* entry, uint64_t present, uint64_t slot_capacity, bool dedicated) {
        n->entry = entry;
        n->slot_capacity = slot_capacity;
        n->dedicated = dedicated;
        link_back(n, present);
    }
    // A bind: `n` moves to the back with this present.
    void touch(Node* n, uint64_t present) {
        unlink(n);
        link_back(n, present);
    }
    void erase(Node* n) { unlink(n); }
    // Forgets every node (their memory is the caller's).
    void clear() noexcept {
        head_ = tail_ = boundary_ = nullptr;
        size_ = 0;
        groups_.clear();
        buckets_ = {};
        nonempty_ = {};
    }

    // The reference walk (see the file comment). *steps = nodes looked at.
    Node* find_reuse_linear(uint64_t present, bool forced, uint64_t need, bool dedicated, uint64_t reuse_after,
                            size_t* steps) const {
        size_t scanned = 0;
        Node* found = nullptr;
        for (Node* l = head_; l != nullptr && scanned < window_; l = l->next_, ++scanned) {
            if (!forced && present < l->last_used_present + reuse_after) {
                break;
            }
            if (l->dedicated == dedicated && l->slot_capacity >= need) {
                found = l;
                break;
            }
        }
        if (steps != nullptr) {
            *steps = scanned;
        }
        return found;
    }

    // Same answer as find_reuse_linear. *steps = groups, buckets and nodes looked at.
    Node* find_reuse(uint64_t present, bool forced, uint64_t need, bool dedicated, uint64_t reuse_after,
                     size_t* steps) const {
        size_t looked = 0;
        // The walk cannot reach nodes at or after `bound`.
        uint64_t bound = boundary_ != nullptr ? boundary_->seq_ : kNoSeq;
        if (!forced) {
            for (auto g = groups_.rbegin(); g != groups_.rend() && present < g->first + reuse_after; ++g) {
                ++looked;
                bound = std::min(bound, g->second.head->seq_);
            }
        }
        Node* best = nullptr;
        uint64_t best_seq = bound;
        const auto& kind = buckets_[dedicated ? 1 : 0];
        const auto& mask = nonempty_[dedicated ? 1 : 0];
        const size_t first = bucket_of(need);
        // Buckets above need's own: any node is large enough, the front is the bucket's oldest.
        const size_t above = first + 1;  // < kBuckets: bucket_of() is at most 251
        for (size_t w = above / 64; w < mask.size(); ++w) {
            uint64_t bits = mask[w];
            if (w == above / 64) {
                bits &= ~uint64_t{0} << (above % 64);
            }
            while (bits != 0) {
                const size_t b = w * 64 + static_cast<size_t>(std::countr_zero(bits));
                bits &= bits - 1;
                ++looked;
                Node* h = kind[b].head;
                if (h->seq_ < best_seq) {
                    best = h;
                    best_seq = h->seq_;
                }
            }
        }
        // need's own bucket: oldest first, up to the best candidate so far.
        for (Node* n = kind[first].head; n != nullptr && n->seq_ < best_seq; n = n->bnext_) {
            ++looked;
            if (n->slot_capacity >= need) {
                best = n;
                best_seq = n->seq_;
                break;
            }
        }
        if (steps != nullptr) {
            *steps = looked;
        }
        return best;
    }

private:
    static constexpr uint64_t kNoSeq = std::numeric_limits<uint64_t>::max();
    static constexpr size_t kBuckets = 256;  // 4 per power of two over 64-bit capacities (252 used)
    struct List {
        Node* head = nullptr;
        Node* tail = nullptr;
    };

    // Monotone in the capacity: [0, 4) one bucket each, then four per power of two.
    static size_t bucket_of(uint64_t cap) noexcept {
        if (cap < 4) {
            return static_cast<size_t>(cap);
        }
        const unsigned e = 63u - static_cast<unsigned>(std::countl_zero(cap));
        return 4u * (e - 1u) + static_cast<size_t>((cap >> (e - 2u)) & 3u);
    }

    void link_back(Node* n, uint64_t present) {
        n->last_used_present = present;
        n->seq_ = ++seq_;
        n->next_ = nullptr;
        n->prev_ = tail_;
        (tail_ != nullptr ? tail_->next_ : head_) = n;
        tail_ = n;
        ++size_;
        if (boundary_ == nullptr && size_ > window_) {
            boundary_ = n;  // size was window_: n is the node at rank window_
        }

        const size_t kind = n->dedicated ? 1 : 0;
        const size_t b = bucket_of(n->slot_capacity);
        List& bl = buckets_[kind][b];
        n->bnext_ = nullptr;
        n->bprev_ = bl.tail;
        (bl.tail != nullptr ? bl.tail->bnext_ : bl.head) = n;
        bl.tail = n;
        nonempty_[kind][b / 64] |= uint64_t{1} << (b % 64);

        // Presents only grow, so the group is nearly always the last one.
        auto g = groups_.empty() || groups_.rbegin()->first != present ? groups_.try_emplace(present).first
                                                                       : std::prev(groups_.end());
        typename Node::PresentGroup& pg = g->second;
        n->group_ = &pg;
        n->gnext_ = nullptr;
        n->gprev_ = pg.tail;
        (pg.tail != nullptr ? pg.tail->gnext_ : pg.head) = n;
        pg.tail = n;
    }

    void unlink(Node* n) {
        if (boundary_ != nullptr && (n == boundary_ || n->seq_ < boundary_->seq_)) {
            boundary_ = boundary_->next_;  // one node before it left the window: the next one moves in
        }
        (n->prev_ != nullptr ? n->prev_->next_ : head_) = n->next_;
        (n->next_ != nullptr ? n->next_->prev_ : tail_) = n->prev_;
        n->prev_ = n->next_ = nullptr;
        --size_;

        const size_t kind = n->dedicated ? 1 : 0;
        const size_t b = bucket_of(n->slot_capacity);
        List& bl = buckets_[kind][b];
        (n->bprev_ != nullptr ? n->bprev_->bnext_ : bl.head) = n->bnext_;
        (n->bnext_ != nullptr ? n->bnext_->bprev_ : bl.tail) = n->bprev_;
        n->bprev_ = n->bnext_ = nullptr;
        if (bl.head == nullptr) {
            nonempty_[kind][b / 64] &= ~(uint64_t{1} << (b % 64));
        }

        typename Node::PresentGroup& pg = *n->group_;
        (n->gprev_ != nullptr ? n->gprev_->gnext_ : pg.head) = n->gnext_;
        (n->gnext_ != nullptr ? n->gnext_->gprev_ : pg.tail) = n->gprev_;
        n->gprev_ = n->gnext_ = nullptr;
        n->group_ = nullptr;
        if (pg.head == nullptr) {
            groups_.erase(n->last_used_present);
        }
    }

    size_t window_;
    Node* head_ = nullptr;
    Node* tail_ = nullptr;
    Node* boundary_ = nullptr;  // node at rank window_, or null while size_ <= window_
    size_t size_ = 0;
    uint64_t seq_ = 0;
    std::map<uint64_t, typename Node::PresentGroup> groups_;  // present -> its nodes, oldest first
    std::array<std::array<List, kBuckets>, 2> buckets_{};     // [dedicated][capacity bucket]
    std::array<std::array<uint64_t, kBuckets / 64>, 2> nonempty_{};
};

}  // namespace playgta6::gpu::d3d12_guest
