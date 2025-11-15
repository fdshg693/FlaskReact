from langgraph.channels import EphemeralValue
from langgraph.pregel import Pregel, NodeBuilder
from langgraph.pregel._write import ChannelWriteEntry


def test1():
    node1 = NodeBuilder().subscribe_only("a").do(lambda x: x + x).write_to("b")
    app = Pregel(
        nodes={"node1": node1},
        channels={
            "a": EphemeralValue(str),
            "b": EphemeralValue(str),
        },
        input_channels=["a"],
        output_channels=["b"],
    )

    result = app.invoke({"a": "foo"})
    print(result["b"])  # Should print "foofoo"


def test2():
    example_node = (
        NodeBuilder()
        .subscribe_only("value")
        .do(lambda x: x + x if len(x) < 10 else None)
        .write_to(ChannelWriteEntry(channel="value", skip_none=True))
    )

    app = Pregel(
        nodes={"example_node": example_node},
        channels={
            "value": EphemeralValue(str),
        },
        input_channels=["value"],
        output_channels=["value"],
    )

    result = app.invoke({"value": "abc"})
    print(result["value"])  # Should print "abcabcabcabc"


if __name__ == "__main__":
    # test1()
    test2()
