"""Seed 10 sample Sadhguru documents for dev/testing.

Usage: python scripts/seed_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pipeline import ingest_documents
from scraper.models import ScrapedDocument

SAMPLE_DOCS = [
    ScrapedDocument(
        title="What Is Karma?",
        url="https://isha.sadhguru.org/en/wisdom/article/what-is-karma",
        content_type="article",
        topics=["karma", "action", "life", "consciousness"],
        content="""Karma literally means action. Action is of three kinds -
it could be in terms of the body, it could be in terms of the mind, or it
could be in terms of your energy. Whatever you do with your body, mind, or
energy, it leaves a certain residual impact. This residual impact is what is
being referred to as karma.

Karma is the memory of life. Every life form carries this memory. It is
because of this that an atom behaves the way it does, a molecule behaves the
way it does, and every life form manifests the way it does. Everything that
you call as "myself" is a certain accumulation of karma - how your body is,
how your mind functions, how your life situations happen, everything is
determined by this.

But the beauty of being human is, you can break the cycle. You can act
consciously instead of compulsively. Once you shift from compulsive action
to conscious action, you start generating the kind of karma that leads to
your liberation, not your bondage.""",
    ),
    ScrapedDocument(
        title="The Significance of Meditation",
        url="https://isha.sadhguru.org/en/wisdom/article/significance-of-meditation",
        content_type="article",
        topics=["meditation", "mind", "consciousness", "stillness", "peace"],
        content="""Meditation is not an act - it is a quality. If you cultivate
your body, your mind, your energies and your emotions to a certain level of
maturity, meditation will naturally happen. It is like a fragrance. If you
cultivate a flower, the fragrance is a natural consequence.

Meditation means to be in a state where you are not doing anything - not
with the body, not with the mind. If you simply sit, you are aware of the
nature of everything around you. This is meditation. When you are aware, the
noise in your mind drops. In awareness, naturally your mind becomes silent.

The whole purpose of meditation is to make you aware that the noise is you
and the silence is also you. Right now, you are identified with the noise.
When you become identified with the silence, you know the nature of your
existence.""",
    ),
    ScrapedDocument(
        title="Understanding Yoga",
        url="https://isha.sadhguru.org/en/wisdom/article/understanding-yoga",
        content_type="article",
        topics=["yoga", "body", "mind", "energy", "union"],
        content="""Yoga means union. The word yoga literally means "that which
brings you to reality." If you experience everything as one in your
consciousness, then you are in yoga. To attain that union within you is yoga.

Yoga is not about bending and twisting your body or holding your breath. It
is a mechanism and a technology to get you to that state of experience where
you see reality just the way it is. If you want to know the experiential
nature of everything that is you, and everything that is around you, you
need a technology - that technology is yoga.

When we say yoga, we do not mean any particular practice or posture. The
entire system of yoga is oriented towards one fundamental goal - that you
experience the ultimate nature of your existence. Everything else is just a
stepping stone towards that.""",
    ),
    ScrapedDocument(
        title="What Happens After Death?",
        url="https://isha.sadhguru.org/en/wisdom/article/what-happens-after-death",
        content_type="article",
        topics=["death", "afterlife", "rebirth", "consciousness", "life"],
        content="""Death is not the opposite of life. Death is the opposite of
birth. Life has no opposite. Life is an eternal process. What you call as
death is simply a transition.

When a person dies, the physical body will go back to the earth. What you
call as the mind - which is a huge volume of memory and information - will
continue. This information body, or what we call the "karmic body," will
carry on and find another womb.

The quality of this transition determines the quality of your next life. If
you live joyfully, if you live consciously, if you live with awareness, the
transition will be smooth and graceful. If you live in fear, anxiety, and
confusion, the transition will be painful and confused.

This is why spiritual practices are important - not just for this life but
for every dimension of your existence, including the transition that you
call death.""",
    ),
    ScrapedDocument(
        title="On Love and Relationships",
        url="https://isha.sadhguru.org/en/wisdom/article/love-and-relationships",
        content_type="article",
        topics=["love", "relationships", "joy", "emotion", "happiness"],
        content="""Love is not something you do. Love is something that you are.
If you are loving by your own nature, you can have any kind of relationship
with anyone without any problem. The problem is, most people are trying to
extract love from another person. When you are trying to extract, it
becomes a painful process.

Love is your quality, not what someone does to you. If you wait for someone
to make you joyful, you will wait forever. Joy is not something that comes
to you from outside - it is what happens when you are in touch with the
deeper dimensions of who you are.

A relationship is not about finding the right person. It is about being the
right person. If two people are truly joyful by their own nature, they can
have a wonderful relationship. But if they are trying to squeeze joy out of
each other, it will become ugly.""",
    ),
    ScrapedDocument(
        title="The Power of Shiva",
        url="https://isha.sadhguru.org/en/wisdom/article/power-of-shiva",
        content_type="article",
        topics=["shiva", "adiyogi", "consciousness", "yoga", "devotion"],
        content="""Shiva is not a god up in the sky. Shiva refers to "that which
is not." Creation is "that which is." The basis of creation is "that which
is not." If you look at the night sky, you see a few stars and a vast
emptiness. This vast emptiness is Shiva.

Shiva is described as Adiyogi, the first yogi. He is the source of yoga.
Over fifteen thousand years ago, he appeared in the upper regions of the
Himalayas and transmitted the yogic sciences to seven sages, the
Saptarishis. These seven sages carried this knowledge across the world.

When we say Shiva, we are talking about the fundamental basis of all
existence. Shiva is not a person or a deity - Shiva is the very space in
which everything exists and into which everything dissolves. To experience
Shiva is to experience the boundless nature of your own being.""",
    ),
    ScrapedDocument(
        title="Inner Engineering: A Path to Joy",
        url="https://isha.sadhguru.org/en/wisdom/article/inner-engineering-path-to-joy",
        content_type="article",
        topics=["inner engineering", "joy", "wellbeing", "sadhana", "kriya"],
        content="""Inner Engineering is a comprehensive system addressing every
aspect of human wellbeing - from body and mind to emotions and energy. It
offers tools and solutions to empower yourself to create your life the way
you want it.

The core of Inner Engineering is Shambhavi Mahamudra Kriya, a powerful 21-
minute practice that brings your entire system into alignment. This is not
something that I invented - this comes from a very ancient yogic tradition
that has been transmitted from guru to disciple for thousands of years.

When I say "engineering," I mean there is a technology to inner wellbeing
just as there is a technology for external wellbeing. You have learned to
manage the outside world reasonably well. Now it is time to turn inward and
manage the inner world. If the outside and the inside are in sync, you will
naturally be joyful.""",
    ),
    ScrapedDocument(
        title="Sadhguru on Conscious Planet",
        url="https://isha.sadhguru.org/en/wisdom/article/conscious-planet",
        content_type="article",
        topics=["conscious planet", "save soil", "environment", "ecology", "earth"],
        content="""If we do not act now, in the next 25 to 30 years, we will have
a soil crisis that will make the water crisis look like a picnic. Eighty-
seven percent of all life forms on this planet - including human beings -
are sustained by the top 39 inches of soil. If this soil is not brought
back to health, we are looking at a catastrophe of unimaginable proportions.

Conscious Planet is a movement to ensure that governments across the world
make soil health a policy priority. It is not about planting a few trees
here and there. It is about fundamentally changing the relationship between
humanity and the soil that sustains us.

Save Soil is not an environmental movement. It is a movement for human
survival. If we exhaust the soil, we exhaust the very basis of our
existence. Every human being on this planet must understand this and act.""",
    ),
    ScrapedDocument(
        title="Sadhguru's Wisdom on Success",
        url="https://isha.sadhguru.org/en/wisdom/article/wisdom-on-success",
        content_type="article",
        topics=["success", "action", "life", "purpose", "commitment"],
        content="""Success does not come to you because you want it. Success
comes to you because you did the right things. You cannot decide the fruit,
but you can decide what kind of seed you plant. If you plant the right seed
and nurture it, the fruit is a natural consequence.

The problem is, most people are not interested in action - they are only
interested in the result. When your focus is on the result, you will never
do your best work. When your focus is on the process, excellence is a
natural consequence.

Don't try to be successful - try to be valuable. If you are truly valuable,
success will chase you. If you are running behind success, you will always
be one step behind. Instead, invest in yourself - in your capabilities, in
your clarity, in your commitment. Everything else will follow.""",
    ),
    ScrapedDocument(
        title="The Nature of Consciousness",
        url="https://isha.sadhguru.org/en/wisdom/article/nature-of-consciousness",
        content_type="article",
        topics=["consciousness", "awareness", "perception", "existence", "being"],
        content="""Consciousness is not something that you create. It is
something that you are. Right now, you are conscious - but you are not
aware that you are conscious. You are busy with the content of your
consciousness rather than the nature of your consciousness.

If you pay attention to the nature of your consciousness rather than its
content, you will see that it has no boundary, no limitation. The problem
with human beings is, they have gotten identified with the limited content
- their body, their mind, their emotions. Because of this identification,
they have forgotten the boundless nature of their consciousness.

Meditation is simply the process of shifting your attention from the
content to the context, from the limited to the unlimited. Once you
experience the unlimited nature of your consciousness, fear and suffering
become irrelevant because they are products of a limited identity.""",
    ),
]


def main() -> None:
    print(f"Seeding {len(SAMPLE_DOCS)} sample documents...")
    count = ingest_documents(SAMPLE_DOCS)
    print(f"Done! {count} chunks indexed into Qdrant.")
    print("\nSeeded topics:")
    all_topics = set()
    for doc in SAMPLE_DOCS:
        all_topics.update(doc.topics)
    for topic in sorted(all_topics):
        print(f"  - {topic}")


if __name__ == "__main__":
    main()
