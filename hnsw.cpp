#include "hnsw.h"

void Layer::Add(size_t index)
{
    if (!bottom_)
    {
        encoder_[index] = size_;
        decoder_[size_] = index;
    }
    ++size_;
    graph_.push_back(std::vector<size_t>());
}

size_t Layer::Encoder(size_t index) const
{
    if (bottom_)
    {
        return index;
    }
    else
    {
        return encoder_.at(index);
    }
}

size_t Layer::Decoder(size_t index) const
{
    if (bottom_)
    {
        return index;
    }
    else
    {
        return decoder_.at(index);
    }
}
