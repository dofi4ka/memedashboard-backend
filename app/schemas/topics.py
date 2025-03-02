from dataclasses import dataclass


@dataclass
class Topic:
    name: str
    chatgpt_description: str
    user_description: str


topics = {
    "котики": Topic(
        name="котики",
        chatgpt_description="Memes about cats featuring humorous situations or expressions, often cute and entertaining. Examples include playful kittens, cats in funny poses, or humorous captions related to cat behavior.",
        user_description="мемы о котиках и их забавных моментах",
    ),
    "милые животные": Topic(
        name="милые животные",
        chatgpt_description="Memes about various cute animals besides cats, such as dogs, rabbits, or other adorable creatures. Examples include funny animal expressions or charming pet moments.",
        user_description="мемы о разных животных, демонстрирующие их очарование и юмор",
    ),
    "социальные ситуации": Topic(
        name="социальные ситуации",
        chatgpt_description="Memes depicting everyday social interactions, online behavior, or humorous takes on society and community dynamics. Examples include memes about social media trends, interactions between people, or ironic commentary on social norms.",
        user_description="мемы о взаимодействии людей в сети и реальной жизни",
    ),
    "повседневные ситуации": Topic(
        name="повседневные ситуации",
        chatgpt_description="Memes about everyday life, including work, routine, stress, or daily challenges. Examples include humorous depictions of office life, commuting woes, or everyday mishaps.",
        user_description="мемы, отражающие жизненные трудности, рабочие будни и бытовые моменты",
    ),
    "игры и развлечения": Topic(
        name="игры и развлечения",
        chatgpt_description="Memes about video games, gaming culture, and entertainment, often including nostalgic or humorous takes on gaming experiences. Examples include funny game references, gaming stereotypes, or playful nods to popular video games.",
        user_description="мемы о видеоиграх, гик-культуре и ностальгических моментах",
    ),
    "кино и сериалы": Topic(
        name="кино и сериалы",
        chatgpt_description="Memes referencing movies, TV shows, or cinematic characters. Examples include quotes or scenes from popular films, parodies of well-known series, or humorous takes on cinematic tropes.",
        user_description="мемы, связанные с фильмами, ТВ-шоу и персонажами",
    ),
    "интернет-культура": Topic(
        name="интернет-культура",
        chatgpt_description="Memes about internet trends, viral content, and online humor. Examples include self-referential memes about meme culture, viral challenges, or internet phenomena.",
        user_description="мемы о самом интернете, вирусных трендах и самоиронии",
    ),
    "праздники": Topic(
        name="праздники",
        chatgpt_description="Memes related to holidays and seasonal events, such as New Year, Christmas, International Women's Day, etc. Examples include holiday-specific humor, festive decorations, or seasonal traditions.",
        user_description="мемы, связанные с сезонными и ежегодными событиями (Новый год, 8 марта, День Победы, Рождество и т.д.)",
    ),
    "политика": Topic(
        name="политика",
        chatgpt_description="Memes about politics, politicians, or political events. Examples include memes about political figures, election campaigns, or political commentary.",
        user_description="мемы о политике, политической жизни и политической культуре",
    ),
    "случайные": Topic(
        name="случайные",
        chatgpt_description="Memes that do not fit into any of the above categories or appear randomly without a clear recurring theme. They are infrequent and diverse in content.",
        user_description="темы которые появляются нерегулярно и не накапливают стабильную популярность",
    ),
}


def validate_topic(topic: str) -> str:
    topic = topic.lower().strip()
    if topic not in topics:
        return "случайные"
    return topic
