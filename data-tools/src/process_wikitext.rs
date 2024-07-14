use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::satisfy,
    combinator::{map, not, opt, recognize, value},
    multi::{many0, many1},
    sequence::{delimited, preceded, terminated},
    IResult,
};

use crate::process_xml::Article;

fn plain(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(
        |c| matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | ' ' | ',' | '.' | '(' | ')' | ':' | '-' | '!' | '/'),
    )))(i)
}

fn link_contents(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(|c| !matches!(c, ']' | '|' | '['))))(i)
}

fn url_contents(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(|c| !matches!(c, ']' | ' ' | '['))))(i)
}

fn image_contents(i: &str) -> IResult<&str, &str> {
    value("", many1(satisfy(|c| !matches!(c, ']' | '['))))(i)
}

fn image_items(i: &str) -> IResult<&str, &str> {
    value("", many1(alt((image_contents, plain_link, named_link))))(i)
}

fn template_contents(i: &str) -> IResult<&str, &str> {
    value("", many1(satisfy(|c| !matches!(c, '}' | '{' | '&'))))(i)
}

fn template_items(i: &str) -> IResult<&str, &str> {
    value(
        "",
        many1(alt((
            comment,
            template_contents,
            template,
            table,
            nowiki,
            reference,
            entity,
            tag("&"),  // e.g. in url's
        ))),
    )(i)
}

fn bold_or_italic(i: &str) -> IResult<&str, &str> {
    value("", alt((tag("'''"), tag("''"))))(i)
}

fn heading(i: &str) -> IResult<&str, &str> {
    value(
        "",
        terminated(preceded(tag("="), many1(tag("="))), opt(tag(" "))),
    )(i)
}

fn entity(i: &str) -> IResult<&str, &str> {
    alt((
        value("&", tag("&amp;")),
        value("<", tag("&lt;")),
        value(">", tag("&gt;")),
        value("\"", tag("&quot;")),
        value(" ", tag("&nbsp;")),
    ))(i)
}

fn nowiki_contents(i: &str) -> IResult<&str, &str> {
    recognize(many0(satisfy(|c| !matches!(c, '&'))))(i)
}

fn nowiki(i: &str) -> IResult<&str, &str> {
    delimited(
        tag("&lt;nowiki&gt;"),
        nowiki_contents,
        tag("&lt;/nowiki&gt;"),
    )(i)
}

fn nolt_contents(i: &str) -> IResult<&str, &str> {
    recognize(many0(satisfy(|c| !matches!(c, '<'))))(i)
}

fn ref_name(i: &str) -> IResult<&str, &str> {
    value(
        "",
        delimited(
            tag("<ref name="),
            many1(satisfy(|c| !matches!(c, '>' | '/'))),
            tag(">"),
        ),
    )(i)
}

fn ref_name_empty(i: &str) -> IResult<&str, &str> {
    value(
        "",
        delimited(
            tag("<ref name="),
            many1(satisfy(|c| !matches!(c, '>' | '/'))),
            tag("/>"),
        ),
    )(i)
}

fn reference(i: &str) -> IResult<&str, &str> {
    alt((
        value("", delimited(tag("<ref>"), nolt_contents, tag("</ref>"))),
        value("", delimited(ref_name, nolt_contents, tag("</ref>"))),
        value(
            "",
            delimited(tag("&lt;ref&gt;"), nowiki_contents, tag("&lt;/ref&gt;")),
        ),
        ref_name_empty,
    ))(i)
}

fn image(i: &str) -> IResult<&str, &str> {
    value(
        "",
        delimited(
            alt((tag("[[image:"), tag("[[Image:"), tag("[[File:"))),
            image_items,
            tag("]]"),
        ),
    )(i)
}

fn plain_link(i: &str) -> IResult<&str, &str> {
    map(delimited(tag("[["), link_contents, tag("]]")), |text| {
        if text.contains(':') {
            ""
        } else {
            text
        }
    })(i)
}

fn named_link(i: &str) -> IResult<&str, &str> {
    let (i, _) = tag("[[")(i)?;
    let (i, _) = link_contents(i)?;
    let (i, _) = tag("|")(i)?;
    let (i, result) = link_contents(i)?;
    let (i, _) = tag("]]")(i)?;
    Ok((i, result))
}

fn url(i: &str) -> IResult<&str, &str> {
    recognize(preceded(
        alt((tag("http://"), tag("https://"))),
        url_contents,
    ))(i)
}

fn url_link(i: &str) -> IResult<&str, &str> {
    let (i, _) = tag("[")(i)?;
    let (i, _) = url(i)?;
    let (i, _) = tag(" ")(i)?;
    let (i, result) = link_contents(i)?;
    let (i, _) = tag("]")(i)?;
    Ok((i, result))
}

fn template(i: &str) -> IResult<&str, &str> {
    value(
        "",
        terminated(
            delimited(tag("{{"), template_items, tag("}}")),
            opt(tag("\n")),
        ),
    )(i)
}

fn table(i: &str) -> IResult<&str, &str> {
    value("", delimited(tag("{|"), template_items, tag("}")))(i)
}

fn comment_char(i: &str) -> IResult<&str, ()> {
    alt((
        value((), satisfy(|c| c != '-')),
        value((), terminated(tag("-"), not(tag("-&gt;")))),
    ))(i)
}

fn comment_char2(i: &str) -> IResult<&str, ()> {
    alt((
        value((), satisfy(|c| c != '-')),
        value((), terminated(tag("-"), not(tag("->")))),
    ))(i)
}

fn comment(i: &str) -> IResult<&str, &str> {
    alt((
        value(
            "",
            delimited(tag("&lt;!--"), many0(comment_char), tag("--&gt;")),
        ),
        value("", delimited(tag("<!--"), many0(comment_char2), tag("-->"))),
    ))(i)
}

fn character(i: &str) -> IResult<&str, &str> {
    recognize(satisfy(|_| true))(i)
}

fn item(i: &str) -> IResult<&str, &str> {
    alt((
        plain,
        bold_or_italic,
        heading,
        nowiki,
        reference,
        entity,
        image,
        plain_link,
        named_link,
        url_link,
        template,
        table,
        comment,
        character,
    ))(i)
}

pub fn strip_wikitext(input: String) -> String {
    let mut result = String::new();
    if let Ok((_, items)) = many0(item)(&input) {
        for &item in &items {
            if !result.is_empty() || item != "\n" {
                // trim initial newline characters
                result.push_str(item);
            }
        }
    }
    result
}

impl Article {
    pub fn strip_wikitext(self) -> Self {
        Article {
            url: self.url,
            text: strip_wikitext(self.text),
            split: self.split,
            tokenize: self.tokenize,
        }
    }
}
